from __future__ import absolute_import, division, print_function
import os
import platform
import re
import stat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_fs_size(self, dev):
    """Get providersize and fragment size and return their product."""
    cmd = self.module.get_bin_path(self.INFO, required=True)
    dummy, out, dummy = self.module.run_command([cmd, str(dev)], check_rc=True, environ_update=self.LANG_ENV)
    fragmentsize = providersize = None
    for line in out.splitlines():
        if line.startswith('fsize'):
            fragmentsize = int(line.split()[1])
        elif 'providersize' in line:
            providersize = int(line.split()[-1])
        if None not in (fragmentsize, providersize):
            break
    else:
        raise ValueError(repr(out))
    return fragmentsize * providersize