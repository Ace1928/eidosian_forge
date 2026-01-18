from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_host_tags(self):
    host_tags = self.module.params.get('host_tags')
    if host_tags is None:
        return None
    return ','.join(host_tags)