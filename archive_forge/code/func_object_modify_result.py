from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
def object_modify_result(self, changed=None, result=None):
    if result is not None:
        self.result['result'] = result
    if changed:
        self.changed()