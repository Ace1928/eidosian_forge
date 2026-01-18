from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def parse_distribution_file_Debian(self, name, data, path, collected_facts):
    debian_facts = {}
    if 'Debian' in data or 'Raspbian' in data:
        debian_facts['distribution'] = 'Debian'
        release = re.search('PRETTY_NAME=[^(]+ \\(?([^)]+?)\\)', data)
        if release:
            debian_facts['distribution_release'] = release.groups()[0]
        if collected_facts['distribution_release'] == 'NA' and 'Debian' in data:
            dpkg_cmd = self.module.get_bin_path('dpkg')
            if dpkg_cmd:
                cmd = "%s --status tzdata|grep Provides|cut -f2 -d'-'" % dpkg_cmd
                rc, out, err = self.module.run_command(cmd)
                if rc == 0:
                    debian_facts['distribution_release'] = out.strip()
        debian_version_path = '/etc/debian_version'
        distdata = get_file_lines(debian_version_path)
        for line in distdata:
            m = re.search('(\\d+)\\.(\\d+)', line.strip())
            if m:
                debian_facts['distribution_minor_version'] = m.groups()[1]
    elif 'Ubuntu' in data:
        debian_facts['distribution'] = 'Ubuntu'
    elif 'SteamOS' in data:
        debian_facts['distribution'] = 'SteamOS'
    elif path in ('/etc/lsb-release', '/etc/os-release') and ('Kali' in data or 'Parrot' in data):
        if 'Kali' in data:
            debian_facts['distribution'] = 'Kali'
        elif 'Parrot' in data:
            debian_facts['distribution'] = 'Parrot'
        release = re.search('DISTRIB_RELEASE=(.*)', data)
        if release:
            debian_facts['distribution_release'] = release.groups()[0]
    elif 'Devuan' in data:
        debian_facts['distribution'] = 'Devuan'
        release = re.search('PRETTY_NAME=\\"?[^(\\"]+ \\(?([^) \\"]+)\\)?', data)
        if release:
            debian_facts['distribution_release'] = release.groups()[0]
        version = re.search('VERSION_ID=\\"(.*)\\"', data)
        if version:
            debian_facts['distribution_version'] = version.group(1)
            debian_facts['distribution_major_version'] = version.group(1)
    elif 'Cumulus' in data:
        debian_facts['distribution'] = 'Cumulus Linux'
        version = re.search('VERSION_ID=(.*)', data)
        if version:
            major, _minor, _dummy_ver = version.group(1).split('.')
            debian_facts['distribution_version'] = version.group(1)
            debian_facts['distribution_major_version'] = major
        release = re.search('VERSION="(.*)"', data)
        if release:
            debian_facts['distribution_release'] = release.groups()[0]
    elif 'Mint' in data:
        debian_facts['distribution'] = 'Linux Mint'
        version = re.search('VERSION_ID=\\"(.*)\\"', data)
        if version:
            debian_facts['distribution_version'] = version.group(1)
            debian_facts['distribution_major_version'] = version.group(1).split('.')[0]
    elif 'UOS' in data or 'Uos' in data or 'uos' in data:
        debian_facts['distribution'] = 'Uos'
        release = re.search('VERSION_CODENAME=\\"?([^\\"]+)\\"?', data)
        if release:
            debian_facts['distribution_release'] = release.groups()[0]
        version = re.search('VERSION_ID=\\"(.*)\\"', data)
        if version:
            debian_facts['distribution_version'] = version.group(1)
            debian_facts['distribution_major_version'] = version.group(1).split('.')[0]
    elif 'Deepin' in data or 'deepin' in data:
        debian_facts['distribution'] = 'Deepin'
        release = re.search('VERSION_CODENAME=\\"?([^\\"]+)\\"?', data)
        if release:
            debian_facts['distribution_release'] = release.groups()[0]
        version = re.search('VERSION_ID=\\"(.*)\\"', data)
        if version:
            debian_facts['distribution_version'] = version.group(1)
            debian_facts['distribution_major_version'] = version.group(1).split('.')[0]
    else:
        return (False, debian_facts)
    return (True, debian_facts)