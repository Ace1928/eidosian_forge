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
def check_service_changed(self):
    if self.state and self.running is None:
        self.module.fail_json(msg='failed determining service state, possible typo of service name?')
    if not self.running and self.state in ['reloaded', 'started']:
        self.svc_change = True
    elif self.running and self.state in ['reloaded', 'stopped']:
        self.svc_change = True
    elif self.state == 'restarted':
        self.svc_change = True
    if self.module.check_mode and self.svc_change:
        self.module.exit_json(changed=True, msg='service state changed')