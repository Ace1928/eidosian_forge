from collections import namedtuple
from textwrap import indent
from numba.types import float32, float64, int16, int32, int64, void, Tuple
from numba.core.typing.templates import signature
def generate_stubs():
    for name, (retty, args) in functions.items():

        def argname(arg):
            if arg.name == 'in':
                return 'x'
            else:
                return arg.name
        argnames = [argname(a) for a in args if not a.is_ptr]
        argstr = ', '.join(argnames)
        signature = create_signature(retty, args)
        param_types = '\n'.join([param_template.format(a=a) for a in args if not a.is_ptr])
        docstring = docstring_template.format(param_types=param_types, retty=signature.return_type, func=name)
        docstring = indent(docstring, '    ')
        print(f'def {name[5:]}({argstr}):\n    """{docstring}"""\n\n')