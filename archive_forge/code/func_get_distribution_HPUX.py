from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def get_distribution_HPUX(self):
    hpux_facts = {}
    rc, out, err = self.module.run_command("/usr/sbin/swlist |egrep 'HPUX.*OE.*[AB].[0-9]+\\.[0-9]+'", use_unsafe_shell=True)
    data = re.search('HPUX.*OE.*([AB].[0-9]+\\.[0-9]+)\\.([0-9]+).*', out)
    if data:
        hpux_facts['distribution_version'] = data.groups()[0]
        hpux_facts['distribution_release'] = data.groups()[1]
    return hpux_facts