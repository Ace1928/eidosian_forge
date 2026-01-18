import cgi
import datetime
import re
from simplegeneric import generic
from wsme.exc import ClientSideError, UnknownArgument, InvalidInput
from wsme.types import iscomplex, list_attributes, Unset
from wsme.types import UserType, ArrayType, DictType, File
from wsme.utils import parse_isodate, parse_isotime, parse_isodatetime
import wsme.runtime
def combine_args(funcdef, akw, allow_override=False):
    newargs, newkwargs = ([], {})
    for args, kwargs in akw:
        for i, arg in enumerate(args):
            n = funcdef.arguments[i].name
            if not allow_override and n in newkwargs:
                raise ClientSideError('Parameter %s was given several times' % n)
            newkwargs[n] = arg
        for name, value in kwargs.items():
            n = str(name)
            if not allow_override and n in newkwargs:
                raise ClientSideError('Parameter %s was given several times' % n)
            newkwargs[n] = value
    return (newargs, newkwargs)