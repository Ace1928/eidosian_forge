from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def max_incremental_sync_size(self):
    if not self.full_sync and self._values['max_incremental_sync_size'] is not None:
        if self._values['__warnings'] is None:
            self._values['__warnings'] = []
        self._values['__warnings'].append([dict(msg='"max_incremental_sync_size has no effect if "full_sync" is not true', version='2.4')])
    if self._values['max_incremental_sync_size'] is None:
        return None
    return int(self._values['max_incremental_sync_size'])