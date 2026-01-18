import uuid
from heat.common import short_id
from heat.tests import common
def test_byte_string_60(self):
    val = 76861433640456465
    byte_string = short_id._to_byte_string(val, 60)
    self.assertEqual(b'\x11\x11\x11\x11\x11\x11\x11\x10', byte_string)