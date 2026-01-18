from __future__ import absolute_import
from .Visitor import CythonTransform
from .ModuleNode import ModuleNode
from .Errors import CompileError
from .UtilityCode import CythonUtilityCode
from .Code import UtilityCode, TempitaUtilityCode
from . import Options
from . import Interpreter
from . import PyrexTypes
from . import Naming
from . import Symtab
def analyse_buffer_options(globalpos, env, posargs, dictargs, defaults=None, need_complete=True):
    """
    Must be called during type analysis, as analyse is called
    on the dtype argument.

    posargs and dictargs should consist of a list and a dict
    of tuples (value, pos). Defaults should be a dict of values.

    Returns a dict containing all the options a buffer can have and
    its value (with the positions stripped).
    """
    if defaults is None:
        defaults = buffer_defaults
    posargs, dictargs = Interpreter.interpret_compiletime_options(posargs, dictargs, type_env=env, type_args=(0, 'dtype'))
    if len(posargs) > buffer_positional_options_count:
        raise CompileError(posargs[-1][1], ERR_BUF_TOO_MANY)
    options = {}
    for name, (value, pos) in dictargs.items():
        if name not in buffer_options:
            raise CompileError(pos, ERR_BUF_OPTION_UNKNOWN % name)
        options[name] = value
    for name, (value, pos) in zip(buffer_options, posargs):
        if name not in buffer_options:
            raise CompileError(pos, ERR_BUF_OPTION_UNKNOWN % name)
        if name in options:
            raise CompileError(pos, ERR_BUF_DUP % name)
        options[name] = value
    for name in buffer_options:
        if name not in options:
            try:
                options[name] = defaults[name]
            except KeyError:
                if need_complete:
                    raise CompileError(globalpos, ERR_BUF_MISSING % name)
    dtype = options.get('dtype')
    if dtype and dtype.is_extension_type:
        raise CompileError(globalpos, ERR_BUF_DTYPE)
    ndim = options.get('ndim')
    if ndim and (not isinstance(ndim, int) or ndim < 0):
        raise CompileError(globalpos, ERR_BUF_NDIM)
    mode = options.get('mode')
    if mode and (not mode in ('full', 'strided', 'c', 'fortran')):
        raise CompileError(globalpos, ERR_BUF_MODE)

    def assert_bool(name):
        x = options.get(name)
        if not isinstance(x, bool):
            raise CompileError(globalpos, ERR_BUF_BOOL % name)
    assert_bool('negative_indices')
    assert_bool('cast')
    return options