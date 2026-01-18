import datetime
import decimal
import json
from simplegeneric import generic
import wsme.exc
import wsme.types
from wsme.types import Unset
import wsme.utils
@fromjson.when_type(wsme.types.ArrayType)
def array_fromjson(datatype, value):
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError('Value not a valid list: %s' % value)
    return [fromjson(datatype.item_type, item) for item in value]