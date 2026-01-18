from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def get_distribution_NetBSD(self):
    netbsd_facts = {}
    platform_release = platform.release()
    netbsd_facts['distribution_release'] = platform_release
    rc, out, dummy = self.module.run_command('/sbin/sysctl -n kern.version')
    match = re.match('NetBSD\\s(\\d+)\\.(\\d+)\\s\\((GENERIC)\\).*', out)
    if match:
        netbsd_facts['distribution_major_version'] = match.group(1)
        netbsd_facts['distribution_version'] = '%s.%s' % match.groups()[:2]
    else:
        netbsd_facts['distribution_major_version'] = platform_release.split('.')[0]
        netbsd_facts['distribution_version'] = platform_release
    return netbsd_facts