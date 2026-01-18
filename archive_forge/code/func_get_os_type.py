from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def get_os_type(self, key=None):
    if self.os_type:
        return self._get_by_key(key, self.zone)
    os_type = self.module.params.get('os_type')
    if not os_type:
        return None
    os_types = self.query_api('listOsTypes')
    if os_types:
        for o in os_types['ostype']:
            if os_type in [o['description'], o['id']]:
                self.os_type = o
                return self._get_by_key(key, self.os_type)
    self.fail_json(msg="OS type '%s' not found" % os_type)