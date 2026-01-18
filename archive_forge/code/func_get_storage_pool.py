from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_storage_pool(self):
    """Retrieve storage pool details from the storage array."""
    storage_pools = list()
    try:
        rc, storage_pools = self.request('storage-systems/%s/storage-pools' % self.ssid)
    except Exception as err:
        self.module.fail_json(msg='Failed to obtain list of storage pools.  Array Id [%s]. Error[%s].' % (self.ssid, to_native(err)))
    pool_detail = [storage_pool for storage_pool in storage_pools if storage_pool['name'] == self.storage_pool_name]
    return pool_detail[0] if pool_detail else dict()