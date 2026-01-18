from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def get_distribution_facts(self):
    distribution_facts = {}
    system = platform.system()
    distribution_facts['distribution'] = system
    distribution_facts['distribution_release'] = platform.release()
    distribution_facts['distribution_version'] = platform.version()
    systems_implemented = ('AIX', 'HP-UX', 'Darwin', 'FreeBSD', 'OpenBSD', 'SunOS', 'DragonFly', 'NetBSD')
    if system in systems_implemented:
        cleanedname = system.replace('-', '')
        distfunc = getattr(self, 'get_distribution_' + cleanedname)
        dist_func_facts = distfunc()
        distribution_facts.update(dist_func_facts)
    elif system == 'Linux':
        distribution_files = DistributionFiles(module=self.module)
        dist_file_facts = distribution_files.process_dist_files()
        distribution_facts.update(dist_file_facts)
    distro = distribution_facts['distribution']
    distribution_facts['os_family'] = self.OS_FAMILY.get(distro, None) or distro
    return distribution_facts