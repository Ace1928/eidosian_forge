from openstack.object_store.v1 import obj
from openstack.tests.unit.cloud import test_object as base_test_object
def test_from_body(self):
    sot = obj.Object.existing(container=self.container, **self.body)
    self.assert_no_calls()
    self.assertEqual(self.container, sot.container)
    self.assertEqual(int(self.body['bytes']), sot.content_length)
    self.assertEqual(self.body['last_modified'], sot.last_modified_at)
    self.assertEqual(self.body['hash'], sot.etag)
    self.assertEqual(self.body['content_type'], sot.content_type)