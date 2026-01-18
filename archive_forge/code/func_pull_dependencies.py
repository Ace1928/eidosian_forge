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
def pull_dependencies(self, nodes: t.Iterable[nodes.Node]) -> None:
    """Find all filter and test names used in the template and
        assign them to variables in the compiled namespace. Checking
        that the names are registered with the environment is done when
        compiling the Filter and Test nodes. If the node is in an If or
        CondExpr node, the check is done at runtime instead.

        .. versionchanged:: 3.0
            Filters and tests in If and CondExpr nodes are checked at
            runtime instead of compile time.
        """
    visitor = DependencyFinderVisitor()
    for node in nodes:
        visitor.visit(node)
    for id_map, names, dependency in ((self.filters, visitor.filters, 'filters'), (self.tests, visitor.tests, 'tests')):
        for name in sorted(names):
            if name not in id_map:
                id_map[name] = self.temporary_identifier()
            self.writeline('try:')
            self.indent()
            self.writeline(f'{id_map[name]} = environment.{dependency}[{name!r}]')
            self.outdent()
            self.writeline('except KeyError:')
            self.indent()
            self.writeline('@internalcode')
            self.writeline(f'def {id_map[name]}(*unused):')
            self.indent()
            self.writeline(f'raise TemplateRuntimeError("No {dependency[:-1]} named {name!r} found.")')
            self.outdent()
            self.outdent()