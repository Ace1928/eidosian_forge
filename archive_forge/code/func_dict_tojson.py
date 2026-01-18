import datetime
import decimal
import json
from simplegeneric import generic
import wsme.exc
import wsme.types
from wsme.types import Unset
import wsme.utils
@tojson.when_type(wsme.types.DictType)
def dict_tojson(datatype, value):
    if value is None:
        return None
    return dict(((tojson(datatype.key_type, item[0]), tojson(datatype.value_type, item[1])) for item in value.items()))