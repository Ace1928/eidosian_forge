import os
import sys
from io import BytesIO
from types import CodeType, FunctionType
import dill
from packaging import version
from .. import config
@pklregister(CodeType)
def save_code(pickler, obj):
    dill._dill.logger.trace(pickler, 'Co: %s', obj)
    co_filename = '' if obj.co_filename.startswith('<') or (len(obj.co_filename.split(os.path.sep)) > 1 and obj.co_filename.split(os.path.sep)[-2].startswith('ipykernel_')) or obj.co_name == '<lambda>' else os.path.basename(obj.co_filename)
    co_firstlineno = 1
    if hasattr(obj, 'co_endlinetable'):
        args = (obj.co_lnotab, obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, co_filename, obj.co_name, obj.co_qualname, co_firstlineno, obj.co_linetable, obj.co_endlinetable, obj.co_columntable, obj.co_exceptiontable, obj.co_freevars, obj.co_cellvars)
    elif hasattr(obj, 'co_exceptiontable'):
        args = (obj.co_lnotab, obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, co_filename, obj.co_name, obj.co_qualname, co_firstlineno, obj.co_linetable, obj.co_exceptiontable, obj.co_freevars, obj.co_cellvars)
    elif hasattr(obj, 'co_linetable'):
        args = (obj.co_lnotab, obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, co_filename, obj.co_name, co_firstlineno, obj.co_linetable, obj.co_freevars, obj.co_cellvars)
    elif hasattr(obj, 'co_posonlyargcount'):
        args = (obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, co_filename, obj.co_name, co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars)
    else:
        args = (obj.co_argcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, co_filename, obj.co_name, co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars)
    pickler.save_reduce(dill._dill._create_code, args, obj=obj)
    dill._dill.logger.trace(pickler, '# Co')
    return