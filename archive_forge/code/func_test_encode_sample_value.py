import base64
import datetime
import decimal
import json
from urllib.parse import urlencode
from wsme.exc import ClientSideError, InvalidInput
from wsme.types import isarray, isdict, isusertype, register_type
from wsme.types import UserType, ArrayType, DictType
from wsme.rest import expose, validate
from wsme.rest.json import fromjson, tojson, parse
import wsme.tests.protocol
from wsme.utils import parse_isodatetime, parse_isotime, parse_isodate
def test_encode_sample_value(self):

    class MyType(object):
        aint = int
        astr = str
    register_type(MyType)
    v = MyType()
    v.aint = 4
    v.astr = 's'
    r = wsme.rest.json.encode_sample_value(MyType, v, True)
    print(r)
    assert r[0] == 'javascript'
    assert r[1] == json.dumps({'aint': 4, 'astr': 's'}, ensure_ascii=False, indent=4, sort_keys=True)