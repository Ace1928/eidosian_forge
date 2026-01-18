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
def test_inline_body(self):
    params = urlencode({'__body__': '{"value": 4}'})
    r = self.app.get('/argtypes/setint.json?' + params)
    print(r)
    assert json.loads(r.text) == 4