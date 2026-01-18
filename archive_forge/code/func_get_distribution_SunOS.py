from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def get_distribution_SunOS(self):
    sunos_facts = {}
    data = get_file_content('/etc/release').splitlines()[0]
    if 'Solaris' in data:
        uname_r = get_uname(self.module, flags=['-r'])
        ora_prefix = ''
        if 'Oracle Solaris' in data:
            data = data.replace('Oracle ', '')
            ora_prefix = 'Oracle '
        sunos_facts['distribution'] = data.split()[0]
        sunos_facts['distribution_version'] = data.split()[1]
        sunos_facts['distribution_release'] = ora_prefix + data
        sunos_facts['distribution_major_version'] = uname_r.split('.')[1].rstrip()
        return sunos_facts
    uname_v = get_uname(self.module, flags=['-v'])
    distribution_version = None
    if 'SmartOS' in data:
        sunos_facts['distribution'] = 'SmartOS'
        if _file_exists('/etc/product'):
            product_data = dict([l.split(': ', 1) for l in get_file_content('/etc/product').splitlines() if ': ' in l])
            if 'Image' in product_data:
                distribution_version = product_data.get('Image').split()[-1]
    elif 'OpenIndiana' in data:
        sunos_facts['distribution'] = 'OpenIndiana'
    elif 'OmniOS' in data:
        sunos_facts['distribution'] = 'OmniOS'
        distribution_version = data.split()[-1]
    elif uname_v is not None and 'NexentaOS_' in uname_v:
        sunos_facts['distribution'] = 'Nexenta'
        distribution_version = data.split()[-1].lstrip('v')
    if sunos_facts.get('distribution', '') in ('SmartOS', 'OpenIndiana', 'OmniOS', 'Nexenta'):
        sunos_facts['distribution_release'] = data.strip()
        if distribution_version is not None:
            sunos_facts['distribution_version'] = distribution_version
        elif uname_v is not None:
            sunos_facts['distribution_version'] = uname_v.splitlines()[0].strip()
        return sunos_facts
    return sunos_facts