import copy
from openstack.block_storage.v3 import block_storage_summary as summary
from openstack.tests.unit import base
def test_get_summary_326(self):
    summary_resource = summary.BlockStorageSummary(**BLOCK_STORAGE_SUMMARY_326)
    self.assertEqual(BLOCK_STORAGE_SUMMARY_326['total_size'], summary_resource.total_size)
    self.assertEqual(BLOCK_STORAGE_SUMMARY_326['total_count'], summary_resource.total_count)
    self.assertEqual(BLOCK_STORAGE_SUMMARY_326['metadata'], summary_resource.metadata)