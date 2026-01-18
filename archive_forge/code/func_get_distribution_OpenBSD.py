from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def get_distribution_OpenBSD(self):
    openbsd_facts = {}
    openbsd_facts['distribution_version'] = platform.release()
    rc, out, err = self.module.run_command('/sbin/sysctl -n kern.version')
    match = re.match('OpenBSD\\s[0-9]+.[0-9]+-(\\S+)\\s.*', out)
    if match:
        openbsd_facts['distribution_release'] = match.groups()[0]
    else:
        openbsd_facts['distribution_release'] = 'release'
    return openbsd_facts