from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def opened_luks_name(self):
    """ If luks is already opened, return its name.
            If 'name' parameter is specified and differs
            from obtained value, fail.
            Return None otherwise
        """
    if self._module.params['state'] != 'opened':
        return None
    name = self._crypthandler.get_container_name_by_device(self.device)
    if name is None:
        return None
    if self._module.params['name'] is None:
        return name
    if name != self._module.params['name']:
        self._module.fail_json(msg="LUKS container is already opened under different name '%s'." % name)
    return name