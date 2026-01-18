from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def percent_check(self, appdirect, memmode, reserved=None):
    if appdirect is None or (appdirect < 0 or appdirect > 100):
        return 'appdirect percent should be from 0 to 100.'
    if memmode is None or (memmode < 0 or memmode > 100):
        return 'memorymode percent should be from 0 to 100.'
    if reserved is None:
        if appdirect + memmode > 100:
            return 'Total percent should be less equal 100.'
    else:
        if reserved < 0 or reserved > 100:
            return 'reserved percent should be from 0 to 100.'
        if appdirect + memmode + reserved != 100:
            return 'Total percent should be 100.'