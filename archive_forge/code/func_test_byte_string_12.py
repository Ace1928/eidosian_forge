import uuid
from heat.common import short_id
from heat.tests import common
def test_byte_string_12(self):
    self.assertEqual(b'\xab\xc0', short_id._to_byte_string(2748, 12))
    self.assertEqual(b'\n\xb0', short_id._to_byte_string(171, 12))