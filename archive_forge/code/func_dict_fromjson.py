import datetime
import decimal
import json
from simplegeneric import generic
import wsme.exc
import wsme.types
from wsme.types import Unset
import wsme.utils
@fromjson.when_type(wsme.types.DictType)
def dict_fromjson(datatype, value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('Value not a valid dict: %s' % value)
    return dict(((fromjson(datatype.key_type, item[0]), fromjson(datatype.value_type, item[1])) for item in value.items()))