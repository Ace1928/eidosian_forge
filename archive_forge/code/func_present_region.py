from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def present_region(self):
    region = self.get_region()
    if not region:
        region = self._create_region(region=region)
    else:
        region = self._update_region(region=region)
    return region