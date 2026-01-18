from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
def proxmox_domain_info_argument_spec():
    return dict(domain=dict(type='str', aliases=['realm', 'name']))