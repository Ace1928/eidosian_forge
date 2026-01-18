import json
from openstack.object_store.v1 import container
from openstack.tests.unit import base
def test_create_and_head(self):
    sot = container.Container(**self.body_plus_headers)
    self.assertEqual(self.body_plus_headers['name'], sot.id)
    self.assertEqual(self.body_plus_headers['name'], sot.name)
    self.assertEqual(self.body_plus_headers['count'], sot.count)
    self.assertEqual(self.body_plus_headers['bytes'], sot.bytes)
    self.assertEqual(int(self.body_plus_headers['x-container-object-count']), sot.object_count)
    self.assertEqual(int(self.body_plus_headers['x-container-bytes-used']), sot.bytes_used)
    self.assertEqual(self.body_plus_headers['x-container-read'], sot.read_ACL)
    self.assertEqual(self.body_plus_headers['x-container-write'], sot.write_ACL)
    self.assertEqual(self.body_plus_headers['x-container-sync-to'], sot.sync_to)
    self.assertEqual(self.body_plus_headers['x-container-sync-key'], sot.sync_key)
    self.assertEqual(self.body_plus_headers['x-versions-location'], sot.versions_location)
    self.assertEqual(self.body_plus_headers['x-history-location'], sot.history_location)
    self.assertEqual(self.body_plus_headers['x-timestamp'], sot.timestamp)
    self.assertEqual(self.body_plus_headers['x-storage-policy'], sot.storage_policy)