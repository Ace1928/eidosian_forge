from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def parse_distribution_file_Slackware(self, name, data, path, collected_facts):
    slackware_facts = {}
    if 'Slackware' not in data:
        return (False, slackware_facts)
    slackware_facts['distribution'] = name
    version = re.findall('\\w+[.]\\w+\\+?', data)
    if version:
        slackware_facts['distribution_version'] = version[0]
    return (True, slackware_facts)