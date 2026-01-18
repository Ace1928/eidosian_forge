from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_storage_tags(self):
    storage_tags = self.module.params.get('storage_tags')
    if storage_tags is None:
        return None
    return ','.join(storage_tags)