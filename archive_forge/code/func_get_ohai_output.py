from __future__ import (absolute_import, division, print_function)
import json
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.namespace import PrefixFactNamespace
from ansible.module_utils.facts.collector import BaseFactCollector
def get_ohai_output(self, module):
    ohai_path = self.find_ohai(module)
    if not ohai_path:
        return None
    rc, out, err = self.run_ohai(module, ohai_path)
    if rc != 0:
        return None
    return out