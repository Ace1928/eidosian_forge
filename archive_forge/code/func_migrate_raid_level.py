from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def migrate_raid_level(self, check_mode=False):
    """Request storage pool raid level migration."""
    needs_migration = self.raid_level != self.pool_detail['raidLevel']
    if needs_migration and self.pool_detail['raidLevel'] == 'raidDiskPool':
        self.module.fail_json(msg='Raid level cannot be changed for disk pools')
    if needs_migration and (not check_mode):
        sp_raid_migrate_req = dict(raidLevel=self.raid_level)
        try:
            rc, resp = self.request('storage-systems/%s/storage-pools/%s/raid-type-migration' % (self.ssid, self.name), data=sp_raid_migrate_req, method='POST')
        except Exception as error:
            self.module.fail_json(msg='Failed to change the raid level of storage pool. Array id [%s].  Error[%s].' % (self.ssid, to_native(error)))
    self.pool_detail = self.storage_pool
    return needs_migration