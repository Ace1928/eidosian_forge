from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def present_network(self):
    network = self.get_physical_network()
    if network:
        network = self._update_network()
    else:
        network = self._create_network()
    return network