from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_network_offering(self):
    if self.network_offering:
        return self.network_offering
    args = {'name': self.module.params.get('name'), 'guestiptype': self.module.params.get('guest_type')}
    no = self.query_api('listNetworkOfferings', **args)
    if no:
        self.network_offering = no['networkoffering'][0]
    return self.network_offering