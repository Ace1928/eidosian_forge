import cgi
import datetime
import re
from simplegeneric import generic
from wsme.exc import ClientSideError, UnknownArgument, InvalidInput
from wsme.types import iscomplex, list_attributes, Unset
from wsme.types import UserType, ArrayType, DictType, File
from wsme.utils import parse_isodate, parse_isotime, parse_isodatetime
import wsme.runtime
@from_params.when_type(DictType)
def dict_from_params(datatype, params, path, hit_paths):
    keys = set()
    r = re.compile('^%s\\[(?P<key>[a-zA-Z0-9_\\.]+)\\]' % re.escape(path))
    for p in params.keys():
        m = r.match(p)
        if m:
            keys.add(from_param(datatype.key_type, m.group('key')))
    if not keys:
        return Unset
    return dict(((key, from_params(datatype.value_type, params, '%s[%s]' % (path, key), hit_paths)) for key in keys))