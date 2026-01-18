from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def luks_remove(self):
    return self.device is not None and self._module.params['state'] == 'absent' and self._crypthandler.is_luks(self.device)