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
def modify_service_state(self):
    if self.svc_change:
        if self.state in ['started']:
            self.action = 'start'
        elif not self.running and self.state == 'reloaded':
            self.action = 'start'
        elif self.state == 'stopped':
            self.action = 'stop'
        elif self.state == 'reloaded':
            self.action = 'reload'
        elif self.state == 'restarted':
            self.action = 'restart'
        if self.module.check_mode:
            self.module.exit_json(changed=True, msg='changing service state')
        return self.service_control()
    else:
        rc = 0
        err = ''
        out = ''
        return (rc, out, err)