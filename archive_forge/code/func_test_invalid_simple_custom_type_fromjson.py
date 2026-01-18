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
def test_invalid_simple_custom_type_fromjson(self):
    value = '2b'
    for ba in (True, False):
        jd = '"%s"' if ba else '{"a": "%s"}'
        try:
            i = parse(jd % value, {'a': CustomInt()}, ba)
            self.assertEqual(i, {'a': 2})
        except ClientSideError as e:
            self.assertIsInstance(e, InvalidInput)
            self.assertEqual(e.fieldname, 'a')
            self.assertEqual(e.value, value)
            self.assertEqual(e.msg, "invalid literal for int() with base 10: '%s'" % value)