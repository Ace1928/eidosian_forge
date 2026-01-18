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
def test_GET_bad_simple_accept(self):
    headers = {'Accept': 'text/plain'}
    res = self.app.get('/crud?ref.id=1', headers=headers, status=406)
    print('Received:', res.body)
    assert res.body == b"Unacceptable Accept type: text/plain not in ['application/json', 'text/javascript', 'application/javascript', 'text/xml']"