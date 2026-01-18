from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.facts.collector import BaseFactCollector
@staticmethod
def is_systemd_managed_offline(module):
    if module.get_bin_path('systemctl'):
        if os.path.islink('/sbin/init') and os.path.basename(os.readlink('/sbin/init')) == 'systemd':
            return True
    return False