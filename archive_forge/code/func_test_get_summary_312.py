import copy
from openstack.block_storage.v3 import block_storage_summary as summary
from openstack.tests.unit import base
def test_get_summary_312(self):
    summary_resource = summary.BlockStorageSummary(**BLOCK_STORAGE_SUMMARY_312)
    self.assertEqual(BLOCK_STORAGE_SUMMARY_312['total_size'], summary_resource.total_size)
    self.assertEqual(BLOCK_STORAGE_SUMMARY_312['total_count'], summary_resource.total_count)