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
def test_invalid_json_body(self):
    r = self.app.post('/argtypes/setint.json', '{"value": 2', headers={'Content-Type': 'application/json'}, expect_errors=True)
    print(r)
    assert r.status_int == 400
    assert json.loads(r.text)['faultstring'] == 'Request is not in valid JSON format'