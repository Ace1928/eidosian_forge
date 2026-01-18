from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def parse_distribution_file_ClearLinux(self, name, data, path, collected_facts):
    clear_facts = {}
    if 'clearlinux' not in name.lower():
        return (False, clear_facts)
    pname = re.search('NAME="(.*)"', data)
    if pname:
        if 'Clear Linux' not in pname.groups()[0]:
            return (False, clear_facts)
        clear_facts['distribution'] = pname.groups()[0]
    version = re.search('VERSION_ID=(.*)', data)
    if version:
        clear_facts['distribution_major_version'] = version.groups()[0]
        clear_facts['distribution_version'] = version.groups()[0]
    release = re.search('ID=(.*)', data)
    if release:
        clear_facts['distribution_release'] = release.groups()[0]
    return (True, clear_facts)