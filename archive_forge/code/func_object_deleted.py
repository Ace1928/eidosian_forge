from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
def object_deleted(self):
    self.changed()
    self.result['result'] = 'Object deleted'