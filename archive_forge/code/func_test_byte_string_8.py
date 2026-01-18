import uuid
from heat.common import short_id
from heat.tests import common
def test_byte_string_8(self):
    self.assertEqual(b'\xab', short_id._to_byte_string(171, 8))
    self.assertEqual(b'\x05', short_id._to_byte_string(5, 8))