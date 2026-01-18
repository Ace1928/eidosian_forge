import sys
from types import CodeType
import six
def get_code_params(code):
    params = []
    if hasattr(code, 'co_posonlyargcount'):
        params.append(code.co_posonlyargcount)
    params.extend([code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize, code.co_flags, code.co_code, code.co_consts, code.co_names, code.co_varnames, code.co_filename, code.co_name])
    if hasattr(code, 'co_qualname'):
        params.append(code.co_qualname)
    params.append(code.co_firstlineno)
    if hasattr(code, 'co_linetable'):
        params.append(code.co_linetable)
    else:
        params.append(code.co_lnotab)
    if hasattr(code, 'co_endlinetable'):
        params.append(code.co_endlinetable)
    if hasattr(code, 'co_columntable'):
        params.append(code.co_columntable)
    if hasattr(code, 'co_exceptiontable'):
        params.append(code.co_exceptiontable)
    params.extend([(), ()])
    return tuple(params)