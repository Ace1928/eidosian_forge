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
def test_invalid_date_fromjson_bodyarg(self):
    jdate = '2015-01-invalid'
    try:
        parse('"%s"' % jdate, {'a': datetime.date}, True)
        assert False
    except Exception as e:
        assert isinstance(e, InvalidInput)
        assert e.fieldname == 'a'
        assert e.value == jdate
        assert e.msg == "'%s' is not a legal date value" % jdate