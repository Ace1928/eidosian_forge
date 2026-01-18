from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.facts.collector import BaseFactCollector
@staticmethod
def is_systemd_managed(module):
    if module.get_bin_path('systemctl'):
        for canary in ['/run/systemd/system/', '/dev/.run/systemd/', '/dev/.systemd/']:
            if os.path.exists(canary):
                return True
    return False