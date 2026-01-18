import datetime
import decimal
import json
from simplegeneric import generic
import wsme.exc
import wsme.types
from wsme.types import Unset
import wsme.utils
def encode_result(value, datatype, **options):
    jsondata = tojson(datatype, value)
    if options.get('nest_result', False):
        jsondata = {options.get('nested_result_attrname', 'result'): jsondata}
    return json.dumps(jsondata)