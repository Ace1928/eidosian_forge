import datetime
import decimal
import json
from simplegeneric import generic
import wsme.exc
import wsme.types
from wsme.types import Unset
import wsme.utils
@tojson.when_object(datetime.date)
def date_tojson(datatype, value):
    if value is None:
        return None
    return value.isoformat()