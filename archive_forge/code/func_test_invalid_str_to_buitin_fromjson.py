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
def test_invalid_str_to_buitin_fromjson(self):
    types = (int, float, bool)
    value = '2a'
    for t in types:
        for ba in (True, False):
            jd = '"%s"' if ba else '{"a": "%s"}'
            try:
                parse(jd % value, {'a': t}, ba)
                assert False, "Value '%s' should not parse correctly for %s." % (value, t)
            except ClientSideError as e:
                self.assertIsInstance(e, InvalidInput)
                self.assertEqual(e.fieldname, 'a')
                self.assertEqual(e.value, value)