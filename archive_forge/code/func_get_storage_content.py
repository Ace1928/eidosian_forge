from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_storage_content(self, node, storage, content=None, vmid=None):
    try:
        return self.proxmox_api.nodes(node).storage(storage).content().get(content=content, vmid=vmid)
    except Exception as e:
        self.module.fail_json(msg='Unable to list content on %s, %s for %s and %s: %s' % (node, storage, content, vmid, e))