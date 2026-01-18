import cgi
import datetime
import re
from simplegeneric import generic
from wsme.exc import ClientSideError, UnknownArgument, InvalidInput
from wsme.types import iscomplex, list_attributes, Unset
from wsme.types import UserType, ArrayType, DictType, File
from wsme.utils import parse_isodate, parse_isotime, parse_isodatetime
import wsme.runtime
def args_from_params(funcdef, params):
    kw = {}
    hit_paths = set()
    for argdef in funcdef.arguments:
        value = from_params(argdef.datatype, params, argdef.name, hit_paths)
        if value is not Unset:
            kw[argdef.name] = value
    paths = set(params.keys())
    unknown_paths = paths - hit_paths
    if '__body__' in unknown_paths:
        unknown_paths.remove('__body__')
    if not funcdef.ignore_extra_args and unknown_paths:
        raise UnknownArgument(', '.join(unknown_paths))
    return ([], kw)