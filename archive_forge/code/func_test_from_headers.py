from openstack.object_store.v1 import obj
from openstack.tests.unit.cloud import test_object as base_test_object
def test_from_headers(self):
    sot = obj.Object.existing(container=self.container, **self.headers)
    self.assert_no_calls()
    self.assertEqual(self.container, sot.container)
    self.assertEqual(int(self.headers['Content-Length']), sot.content_length)
    self.assertEqual(self.headers['Accept-Ranges'], sot.accept_ranges)
    self.assertEqual(self.headers['Last-Modified'], sot.last_modified_at)
    self.assertEqual(self.headers['Etag'], sot.etag)
    self.assertEqual(self.headers['X-Timestamp'], sot.timestamp)
    self.assertEqual(self.headers['Content-Type'], sot.content_type)
    self.assertEqual(self.headers['X-Delete-At'], sot.delete_at)
    sot._set_metadata(headers={'x-object-meta-foo': 'bar'})
    self.assert_no_calls()
    self.assertEqual('bar', sot.metadata['foo'])