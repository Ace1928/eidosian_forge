from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec,
from re import compile, match, sub
from time import sleep
def get_create_attributes(self):
    params = dict(((k, v) for k, v in self.module.params.items() if v is not None and k in self.create_update_fields))
    params.update(dict(((k, int(v)) for k, v in params.items() if isinstance(v, bool))))
    return params