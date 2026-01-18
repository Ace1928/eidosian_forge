import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def getcallprotoargument(rout, cb_map={}):
    r = getmultilineblock(rout, 'callprotoargument', comment=0)
    if r:
        return r
    if hascallstatement(rout):
        outmess('warning: callstatement is defined without callprotoargument\n')
        return
    from .capi_maps import getctype
    arg_types, arg_types2 = ([], [])
    if l_and(isstringfunction, l_not(isfunction_wrap))(rout):
        arg_types.extend(['char*', 'size_t'])
    for n in rout['args']:
        var = rout['vars'][n]
        if isintent_callback(var):
            continue
        if n in cb_map:
            ctype = cb_map[n] + '_typedef'
        else:
            ctype = getctype(var)
            if l_and(isintent_c, l_or(isscalar, iscomplex))(var):
                pass
            elif isstring(var):
                pass
            elif not isattr_value(var):
                ctype = ctype + '*'
            if isstring(var) or isarrayofstrings(var) or isstringarray(var):
                arg_types2.append('size_t')
        arg_types.append(ctype)
    proto_args = ','.join(arg_types + arg_types2)
    if not proto_args:
        proto_args = 'void'
    return proto_args