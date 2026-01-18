from __future__ import absolute_import, division, print_function
import glob
import json
import os
import platform
import re
import select
import shlex
import subprocess
import tempfile
import time
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.service import fail_if_missing
from ansible.module_utils.six import PY2, b
def service_enable(self):
    rc, stdout, stderr = self.execute_command('%s -l %s' % (self.svcs_cmd, self.name))
    if rc != 0:
        if stderr:
            self.module.fail_json(msg=stderr)
        else:
            self.module.fail_json(msg=stdout)
    enabled = False
    temporary = False
    for line in stdout.split('\n'):
        if line.startswith('enabled'):
            if 'true' in line:
                enabled = True
            if 'temporary' in line:
                temporary = True
    startup_enabled = enabled and (not temporary) or (not enabled and temporary)
    if self.enable and startup_enabled:
        return
    elif not self.enable and (not startup_enabled):
        return
    if not self.module.check_mode:
        if self.enable:
            subcmd = 'enable -rs'
        else:
            subcmd = 'disable -s'
        rc, stdout, stderr = self.execute_command('%s %s %s' % (self.svcadm_cmd, subcmd, self.name))
        if rc != 0:
            if stderr:
                self.module.fail_json(msg=stderr)
            else:
                self.module.fail_json(msg=stdout)
    self.changed = True