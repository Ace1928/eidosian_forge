import datetime
import decimal
import json
from simplegeneric import generic
import wsme.exc
import wsme.types
from wsme.types import Unset
import wsme.utils
def encode_sample_params(params, format=False):
    kw = {}
    for name, datatype, value in params:
        kw[name] = tojson(datatype, value)
    content = json.dumps(kw, ensure_ascii=False, indent=4 if format else 0, sort_keys=format)
    return ('javascript', content)