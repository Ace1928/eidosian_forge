import uuid
from heat.common import short_id
from heat.tests import common
def test_byte_string_16(self):
    self.assertEqual(b'\xab\xcd', short_id._to_byte_string(43981, 16))
    self.assertEqual(b'\n\xbc', short_id._to_byte_string(2748, 16))