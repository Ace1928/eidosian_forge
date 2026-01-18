import uuid
from oslotest import base as test_base
from oslo_utils import uuidutils
def test_is_uuid_like_insensitive(self):
    self.assertTrue(uuidutils.is_uuid_like(str(uuid.uuid4()).upper()))