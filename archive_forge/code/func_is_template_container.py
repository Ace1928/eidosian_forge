from __future__ import absolute_import, division, print_function
import re
import time
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.proxmox import (
def is_template_container(self, node, vmid):
    """Check if the specified container is a template."""
    proxmox_node = self.proxmox_api.nodes(node)
    config = getattr(proxmox_node, VZ_TYPE)(vmid).config.get()
    return config.get('template', False)