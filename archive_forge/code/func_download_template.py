from __future__ import absolute_import, division, print_function
import os
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def download_template(self, node, storage, template, timeout):
    try:
        taskid = self.proxmox_api.nodes(node).aplinfo.post(storage=storage, template=template)
        return self.task_status(node, taskid, timeout)
    except Exception as e:
        self.module.fail_json(msg='Downloading template %s failed with error: %s' % (template, e))