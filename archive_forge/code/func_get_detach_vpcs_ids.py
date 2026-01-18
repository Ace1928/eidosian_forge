from __future__ import absolute_import, division, print_function
import base64
from .vultr_v2 import AnsibleVultr
def get_detach_vpcs_ids(self, resource, api_version='v1'):
    detach_vpc_ids = []
    for vpc in resource.get(self.VPC_CONFIGS[api_version]['param'], list()):
        param = 'attach_vpc%s' % self.VPC_CONFIGS[api_version]['suffix']
        if vpc['id'] not in list(self.module.params[param]):
            detach_vpc_ids.append(vpc['id'])
    return detach_vpc_ids