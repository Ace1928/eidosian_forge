from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_nextvmid(self):
    try:
        return self.proxmox_api.cluster.nextid.get()
    except Exception as e:
        self.module.fail_json(msg='Unable to retrieve next free vmid: %s' % e)