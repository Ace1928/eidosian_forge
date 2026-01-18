from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def parse_distribution_file_NA(self, name, data, path, collected_facts):
    na_facts = {}
    for line in data.splitlines():
        distribution = re.search('^NAME=(.*)', line)
        if distribution and name == 'NA':
            na_facts['distribution'] = distribution.group(1).strip('"')
        version = re.search('^VERSION=(.*)', line)
        if version and collected_facts['distribution_version'] == 'NA':
            na_facts['distribution_version'] = version.group(1).strip('"')
    return (True, na_facts)