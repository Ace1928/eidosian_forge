from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def delete_storage_pool(self):
    """Delete storage pool."""
    storage_pool_drives = [drive['id'] for drive in self.storage_pool_drives if drive['fdeEnabled']]
    try:
        delete_volumes_parameter = '?delete-volumes=true' if self.remove_volumes else ''
        rc, resp = self.request('storage-systems/%s/storage-pools/%s%s' % (self.ssid, self.pool_detail['id'], delete_volumes_parameter), method='DELETE')
    except Exception as error:
        self.module.fail_json(msg='Failed to delete storage pool. Pool id [%s]. Array id [%s].  Error[%s].' % (self.pool_detail['id'], self.ssid, to_native(error)))
    if storage_pool_drives and self.erase_secured_drives:
        try:
            rc, resp = self.request('storage-systems/%s/symbol/reprovisionDrive?verboseErrorResponse=true' % self.ssid, method='POST', data=dict(driveRef=storage_pool_drives))
        except Exception as error:
            self.module.fail_json(msg='Failed to erase drives prior to creating new storage pool. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))