import typing as t
from contextlib import contextmanager
from functools import update_wrapper
from io import StringIO
from itertools import chain
from keyword import iskeyword as is_python_keyword
from markupsafe import escape
from markupsafe import Markup
from . import nodes
from .exceptions import TemplateAssertionError
from .idtracking import Symbols
from .idtracking import VAR_LOAD_ALIAS
from .idtracking import VAR_LOAD_PARAMETER
from .idtracking import VAR_LOAD_RESOLVE
from .idtracking import VAR_LOAD_UNDEFINED
from .nodes import EvalContext
from .optimizer import Optimizer
from .utils import _PassArg
from .utils import concat
from .visitor import NodeVisitor
def macro_body(self, node: t.Union[nodes.Macro, nodes.CallBlock], frame: Frame) -> t.Tuple[Frame, MacroRef]:
    """Dump the function def of a macro or call block."""
    frame = frame.inner()
    frame.symbols.analyze_node(node)
    macro_ref = MacroRef(node)
    explicit_caller = None
    skip_special_params = set()
    args = []
    for idx, arg in enumerate(node.args):
        if arg.name == 'caller':
            explicit_caller = idx
        if arg.name in ('kwargs', 'varargs'):
            skip_special_params.add(arg.name)
        args.append(frame.symbols.ref(arg.name))
    undeclared = find_undeclared(node.body, ('caller', 'kwargs', 'varargs'))
    if 'caller' in undeclared:
        if explicit_caller is not None:
            try:
                node.defaults[explicit_caller - len(node.args)]
            except IndexError:
                self.fail('When defining macros or call blocks the special "caller" argument must be omitted or be given a default.', node.lineno)
        else:
            args.append(frame.symbols.declare_parameter('caller'))
        macro_ref.accesses_caller = True
    if 'kwargs' in undeclared and 'kwargs' not in skip_special_params:
        args.append(frame.symbols.declare_parameter('kwargs'))
        macro_ref.accesses_kwargs = True
    if 'varargs' in undeclared and 'varargs' not in skip_special_params:
        args.append(frame.symbols.declare_parameter('varargs'))
        macro_ref.accesses_varargs = True
    frame.require_output_check = False
    frame.symbols.analyze_node(node)
    self.writeline(f'{self.func('macro')}({', '.join(args)}):', node)
    self.indent()
    self.buffer(frame)
    self.enter_frame(frame)
    self.push_parameter_definitions(frame)
    for idx, arg in enumerate(node.args):
        ref = frame.symbols.ref(arg.name)
        self.writeline(f'if {ref} is missing:')
        self.indent()
        try:
            default = node.defaults[idx - len(node.args)]
        except IndexError:
            self.writeline(f'{ref} = undefined("parameter {arg.name!r} was not provided", name={arg.name!r})')
        else:
            self.writeline(f'{ref} = ')
            self.visit(default, frame)
        self.mark_parameter_stored(ref)
        self.outdent()
    self.pop_parameter_definitions()
    self.blockvisit(node.body, frame)
    self.return_buffer_contents(frame, force_unescaped=True)
    self.leave_frame(frame, with_python_scope=True)
    self.outdent()
    return (frame, macro_ref)