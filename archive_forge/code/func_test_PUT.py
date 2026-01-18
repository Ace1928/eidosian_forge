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
def test_PUT(self):
    data = {'id': 1, 'name': 'test'}
    content = json.dumps(dict(data=data))
    headers = {'Content-Type': 'application/json'}
    res = self.app.put('/crud', content, headers=headers, expect_errors=False)
    print('Received:', res.body)
    result = json.loads(res.text)
    print(result)
    assert result['data']['id'] == 1
    assert result['data']['name'] == 'test'
    assert result['message'] == 'create'