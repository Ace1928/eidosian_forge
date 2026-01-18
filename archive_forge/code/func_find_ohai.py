from __future__ import (absolute_import, division, print_function)
import json
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.namespace import PrefixFactNamespace
from ansible.module_utils.facts.collector import BaseFactCollector
def find_ohai(self, module):
    ohai_path = module.get_bin_path('ohai')
    return ohai_path