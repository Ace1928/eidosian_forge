from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_nic_from_result(self, result):
    for nic in result.get('nic') or []:
        if nic['networkid'] == self.get_network(key='id'):
            return nic