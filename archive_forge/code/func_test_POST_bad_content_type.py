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
def test_POST_bad_content_type(self):
    headers = {'Content-Type': 'text/plain'}
    res = self.app.post('/crud', json.dumps(dict(data=dict(id=1, name='test'))), headers=headers, status=415)
    print('Received:', res.body)
    assert res.body == b"Unacceptable Content-Type: text/plain not in ['application/json', 'text/javascript', 'application/javascript', 'text/xml']"