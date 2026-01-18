from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
@property
def storage_pool_volumes(self):
    """Retrieve list of volumes associated with storage pool."""
    volumes_resp = None
    try:
        rc, volumes_resp = self.request('storage-systems/%s/volumes' % self.ssid)
    except Exception as err:
        self.module.fail_json(msg='Failed to get storage pools. Array id [%s]. Error[%s]. State[%s].' % (self.ssid, to_native(err), self.state))
    group_ref = self.storage_pool['volumeGroupRef']
    storage_pool_volume_list = [volume['id'] for volume in volumes_resp if volume['volumeGroupRef'] == group_ref]
    return storage_pool_volume_list