from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def get_disk_offering(self, key=None):
    disk_offering = self.module.params.get('disk_offering')
    if not disk_offering:
        return None
    disk_offerings = self.query_api('listDiskOfferings')
    if disk_offerings:
        for d in disk_offerings['diskoffering']:
            if disk_offering in [d['displaytext'], d['name'], d['id']]:
                return self._get_by_key(key, d)
    self.fail_json(msg="Disk offering '%s' not found" % disk_offering)