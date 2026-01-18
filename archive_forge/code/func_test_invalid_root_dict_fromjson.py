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
def test_invalid_root_dict_fromjson(self):
    try:
        parse('["invalid"]', {'a': ArrayType(str)}, False)
        assert False
    except Exception as e:
        assert isinstance(e, ClientSideError)
        assert e.msg == 'Request must be a JSON dict'