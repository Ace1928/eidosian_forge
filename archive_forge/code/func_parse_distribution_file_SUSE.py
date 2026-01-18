from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def parse_distribution_file_SUSE(self, name, data, path, collected_facts):
    suse_facts = {}
    if 'suse' not in data.lower():
        return (False, suse_facts)
    if path == '/etc/os-release':
        for line in data.splitlines():
            distribution = re.search('^NAME=(.*)', line)
            if distribution:
                suse_facts['distribution'] = distribution.group(1).strip('"')
            distribution_version = re.search('^VERSION_ID="?([0-9]+\\.?[0-9]*)"?', line)
            if distribution_version:
                suse_facts['distribution_version'] = distribution_version.group(1)
                suse_facts['distribution_major_version'] = distribution_version.group(1).split('.')[0]
            if 'open' in data.lower():
                release = re.search('^VERSION_ID="?[0-9]+\\.?([0-9]*)"?', line)
                if release:
                    suse_facts['distribution_release'] = release.groups()[0]
            elif 'enterprise' in data.lower() and 'VERSION_ID' in line:
                release = re.search('^VERSION_ID="?[0-9]+\\.?([0-9]*)"?', line)
                if release.group(1):
                    release = release.group(1)
                else:
                    release = '0'
                suse_facts['distribution_release'] = release
    elif path == '/etc/SuSE-release':
        if 'open' in data.lower():
            data = data.splitlines()
            distdata = get_file_content(path).splitlines()[0]
            suse_facts['distribution'] = distdata.split()[0]
            for line in data:
                release = re.search('CODENAME *= *([^\n]+)', line)
                if release:
                    suse_facts['distribution_release'] = release.groups()[0].strip()
        elif 'enterprise' in data.lower():
            lines = data.splitlines()
            distribution = lines[0].split()[0]
            if 'Server' in data:
                suse_facts['distribution'] = 'SLES'
            elif 'Desktop' in data:
                suse_facts['distribution'] = 'SLED'
            for line in lines:
                release = re.search('PATCHLEVEL = ([0-9]+)', line)
                if release:
                    suse_facts['distribution_release'] = release.group(1)
                    suse_facts['distribution_version'] = collected_facts['distribution_version'] + '.' + release.group(1)
    if os.path.islink('/etc/products.d/baseproduct') and os.path.realpath('/etc/products.d/baseproduct').endswith('SLES_SAP.prod'):
        suse_facts['distribution'] = 'SLES_SAP'
    return (True, suse_facts)