from __future__ import (absolute_import, division, print_function)
import sys
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.facts.network.base import NetworkCollector
def findstr(self, text, match):
    for line in text.splitlines():
        if match in line:
            found = line
    return found