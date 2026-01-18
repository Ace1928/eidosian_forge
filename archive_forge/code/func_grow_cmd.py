from __future__ import absolute_import, division, print_function
import os
import platform
import re
import stat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def grow_cmd(self, target):
    """Build and return the resizefs commandline as list."""
    cmdline = [self.module.get_bin_path(self.GROW, required=True)]
    cmdline += self.GROW_MAX_SPACE_FLAGS + [target]
    return cmdline