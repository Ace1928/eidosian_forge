from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def parse_distribution_file_Coreos(self, name, data, path, collected_facts):
    coreos_facts = {}
    distro = get_distribution()
    if distro.lower() == 'coreos':
        if not data:
            return (False, coreos_facts)
        release = re.search('^GROUP=(.*)', data)
        if release:
            coreos_facts['distribution_release'] = release.group(1).strip('"')
    else:
        return (False, coreos_facts)
    return (True, coreos_facts)