from __future__ import absolute_import, division, print_function
import json
import re
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native
def get_missing_ro_attributes(self):
    return list(set(self.readonly_attrs) - set(self.get_actual_ro_attributes().keys()))