import uuid
from oslotest import base as test_base
from oslo_utils import uuidutils
def test_is_uuid_like(self):
    self.assertTrue(uuidutils.is_uuid_like(str(uuid.uuid4())))
    self.assertTrue(uuidutils.is_uuid_like('{12345678-1234-5678-1234-567812345678}'))
    self.assertTrue(uuidutils.is_uuid_like('12345678123456781234567812345678'))
    self.assertTrue(uuidutils.is_uuid_like('urn:uuid:12345678-1234-5678-1234-567812345678'))
    self.assertTrue(uuidutils.is_uuid_like('urn:bbbaaaaa-aaaa-aaaa-aabb-bbbbbbbbbbbb'))
    self.assertTrue(uuidutils.is_uuid_like('uuid:bbbaaaaa-aaaa-aaaa-aabb-bbbbbbbbbbbb'))
    self.assertTrue(uuidutils.is_uuid_like('{}---bbb---aaa--aaa--aaa-----aaa---aaa--bbb-bbb---bbb-bbb-bb-{}'))