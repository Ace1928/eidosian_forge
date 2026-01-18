from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def get_or_fallback(self, key=None, fallback_key=None):
    value = self.module.params.get(key)
    if not value:
        value = self.module.params.get(fallback_key)
    return value