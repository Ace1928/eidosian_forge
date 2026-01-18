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
def get_sunos_svcs_status(self):
    rc, stdout, stderr = self.execute_command('%s %s' % (self.svcs_cmd, self.name))
    if rc == 1:
        if stderr:
            self.module.fail_json(msg=stderr)
        else:
            self.module.fail_json(msg=stdout)
    lines = stdout.rstrip('\n').split('\n')
    status = lines[-1].split(' ')[0]
    return status