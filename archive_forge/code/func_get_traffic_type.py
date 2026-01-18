from __future__ import absolute_import, division, print_function
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec, cs_required_together
from ansible.module_utils.basic import AnsibleModule
def get_traffic_type(self):
    if self.traffic_type:
        return self.traffic_type
    traffic_type = self.module.params.get('traffic_type')
    traffic_types = self.get_traffic_types()
    if traffic_types:
        for t_type in traffic_types['traffictype']:
            if traffic_type.lower() in [t_type['traffictype'].lower(), t_type['id']]:
                self.traffic_type = t_type
                break
    return self.traffic_type