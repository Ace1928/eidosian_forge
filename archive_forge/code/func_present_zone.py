from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def present_zone(self):
    zone = self.get_zone()
    if zone:
        zone = self._update_zone()
    else:
        zone = self._create_zone()
    return zone