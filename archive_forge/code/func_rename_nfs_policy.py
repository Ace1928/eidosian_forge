from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def rename_nfs_policy(module, blade):
    """Rename NFS Export Policy"""
    changed = True
    if not module.check_mode:
        res = blade.patch_nfs_export_policies(names=[module.params['name']], policy=NfsExportPolicy(name=module.params['rename']))
        if res.status_code != 200:
            module.fail_json(msg='Failed to rename NFS export policy {0} to {1}. Error: {2}'.format(module.params['name'], module.params['rename'], res.errors[0].message))
        module.exit_json(changed=changed)