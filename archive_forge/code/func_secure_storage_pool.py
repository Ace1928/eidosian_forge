from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def secure_storage_pool(self, check_mode=False):
    """Enable security on an existing storage pool"""
    self.pool_detail = self.storage_pool
    needs_secure_pool = False
    if not self.secure_pool and self.pool_detail['securityType'] == 'enabled':
        self.module.fail_json(msg='It is not possible to disable storage pool security! See array documentation.')
    if self.secure_pool and self.pool_detail['securityType'] != 'enabled':
        needs_secure_pool = True
    if needs_secure_pool and (not check_mode):
        try:
            rc, resp = self.request('storage-systems/%s/storage-pools/%s' % (self.ssid, self.pool_detail['id']), data=dict(securePool=True), method='POST')
        except Exception as error:
            self.module.fail_json(msg='Failed to secure storage pool. Pool id [%s]. Array [%s]. Error [%s].' % (self.pool_detail['id'], self.ssid, to_native(error)))
    self.pool_detail = self.storage_pool
    return needs_secure_pool