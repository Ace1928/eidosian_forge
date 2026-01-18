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
def get_service_tools(self):
    self.lssrc_cmd = self.module.get_bin_path('lssrc', True)
    if not self.lssrc_cmd:
        self.module.fail_json(msg='unable to find lssrc binary')
    self.startsrc_cmd = self.module.get_bin_path('startsrc', True)
    if not self.startsrc_cmd:
        self.module.fail_json(msg='unable to find startsrc binary')
    self.stopsrc_cmd = self.module.get_bin_path('stopsrc', True)
    if not self.stopsrc_cmd:
        self.module.fail_json(msg='unable to find stopsrc binary')
    self.refresh_cmd = self.module.get_bin_path('refresh', True)
    if not self.refresh_cmd:
        self.module.fail_json(msg='unable to find refresh binary')