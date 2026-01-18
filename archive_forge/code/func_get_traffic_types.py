from __future__ import absolute_import, division, print_function
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec, cs_required_together
from ansible.module_utils.basic import AnsibleModule
def get_traffic_types(self):
    args = {'physicalnetworkid': self.get_physical_network(key='id')}
    traffic_types = self.query_api('listTrafficTypes', **args)
    return traffic_types