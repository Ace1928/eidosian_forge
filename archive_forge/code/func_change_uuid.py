from __future__ import absolute_import, division, print_function
import os
import platform
import re
import stat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def change_uuid(self, new_uuid, dev):
    """Change filesystem UUID. Returns stdout of used command"""
    if self.module.check_mode:
        self.module.exit_json(change=True, msg='Changing %s filesystem UUID on device %s' % (self.fstype, dev))
    dummy, out, dummy = self.module.run_command(self.change_uuid_cmd(new_uuid=new_uuid, target=str(dev)), check_rc=True)
    return out