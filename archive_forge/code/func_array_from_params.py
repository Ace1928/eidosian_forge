import cgi
import datetime
import re
from simplegeneric import generic
from wsme.exc import ClientSideError, UnknownArgument, InvalidInput
from wsme.types import iscomplex, list_attributes, Unset
from wsme.types import UserType, ArrayType, DictType, File
from wsme.utils import parse_isodate, parse_isotime, parse_isodatetime
import wsme.runtime
@from_params.when_type(ArrayType)
def array_from_params(datatype, params, path, hit_paths):
    if hasattr(params, 'getall'):

        def getall(params, path):
            return params.getall(path)
    elif hasattr(params, 'getlist'):

        def getall(params, path):
            return params.getlist(path)
    if path in params:
        hit_paths.add(path)
        return [from_param(datatype.item_type, value) for value in getall(params, path)]
    if iscomplex(datatype.item_type):
        attributes = set()
        r = re.compile('^%s\\.(?P<attrname>[^\\.])' % re.escape(path))
        for p in params.keys():
            m = r.match(p)
            if m:
                attributes.add(m.group('attrname'))
        if attributes:
            value = []
            for attrdef in list_attributes(datatype.item_type):
                attrpath = '%s.%s' % (path, attrdef.key)
                hit_paths.add(attrpath)
                attrvalues = getall(params, attrpath)
                if len(value) < len(attrvalues):
                    value[-1:] = [datatype.item_type() for i in range(len(attrvalues) - len(value))]
                for i, attrvalue in enumerate(attrvalues):
                    setattr(value[i], attrdef.key, from_param(attrdef.datatype, attrvalue))
            return value
    indexes = set()
    r = re.compile('^%s\\[(?P<index>\\d+)\\]' % re.escape(path))
    for p in params.keys():
        m = r.match(p)
        if m:
            indexes.add(int(m.group('index')))
    if not indexes:
        return Unset
    indexes = list(indexes)
    indexes.sort()
    return [from_params(datatype.item_type, params, '%s[%s]' % (path, index), hit_paths) for index in indexes]