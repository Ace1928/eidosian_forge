from __future__ import (absolute_import, division, print_function)
import os
import subprocess
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector
def pkg_mgrs(self, collected_facts):
    if collected_facts['ansible_os_family'] == 'Altlinux':
        return filter(lambda pkg: pkg['path'] != '/usr/bin/pkg', PKG_MGRS)
    else:
        return PKG_MGRS