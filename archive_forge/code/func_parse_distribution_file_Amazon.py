from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def parse_distribution_file_Amazon(self, name, data, path, collected_facts):
    amazon_facts = {}
    if 'Amazon' not in data:
        return (False, amazon_facts)
    amazon_facts['distribution'] = 'Amazon'
    if path == '/etc/os-release':
        version = re.search('VERSION_ID=\\"(.*)\\"', data)
        if version:
            distribution_version = version.group(1)
            amazon_facts['distribution_version'] = distribution_version
            version_data = distribution_version.split('.')
            if len(version_data) > 1:
                major, minor = version_data
            else:
                major, minor = (version_data[0], 'NA')
            amazon_facts['distribution_major_version'] = major
            amazon_facts['distribution_minor_version'] = minor
    else:
        version = [n for n in data.split() if n.isdigit()]
        version = version[0] if version else 'NA'
        amazon_facts['distribution_version'] = version
    return (True, amazon_facts)