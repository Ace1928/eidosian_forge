import uuid
from oslotest import base as test_base
from oslo_utils import uuidutils
def test_generate_uuid_dashed_false(self):
    uuid_string = uuidutils.generate_uuid(dashed=False)
    self.assertIsInstance(uuid_string, str)
    self.assertEqual(len(uuid_string), 32)
    self.assertNotIn('-', uuid_string)