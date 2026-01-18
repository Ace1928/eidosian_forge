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
def check_systemd():
    if location.get('systemctl', False):
        for canary in ['/run/systemd/system/', '/dev/.run/systemd/', '/dev/.systemd/']:
            if os.path.exists(canary):
                return True
        try:
            f = open('/proc/1/comm', 'r')
        except IOError:
            return False
        for line in f:
            if 'systemd' in line:
                return True
    return False