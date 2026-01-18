from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def luks_close(self):
    if self._module.params['name'] is None and self.device is None or self._module.params['state'] != 'closed':
        return False
    if self.device is not None:
        name = self._crypthandler.get_container_name_by_device(self.device)
        luks_is_open = name is not None
    if self._module.params['name'] is not None:
        self.device = self._crypthandler.get_container_device_by_name(self._module.params['name'])
        luks_is_open = self.device is not None
    return luks_is_open