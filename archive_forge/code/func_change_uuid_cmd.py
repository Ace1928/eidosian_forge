from __future__ import absolute_import, division, print_function
import os
import platform
import re
import stat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def change_uuid_cmd(self, new_uuid, target):
    """Build and return the UUID change command line as list."""
    cmdline = [self.module.get_bin_path(self.CHANGE_UUID, required=True)]
    if self.CHANGE_UUID_OPTION_HAS_ARG:
        cmdline += [self.CHANGE_UUID_OPTION, new_uuid, target]
    else:
        cmdline += [self.CHANGE_UUID_OPTION, target]
    return cmdline