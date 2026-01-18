from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
from tempfile import NamedTemporaryFile
from datetime import datetime
def has_rule(self, rule_type, rule_control, rule_path):
    if self.get(rule_type, rule_control, rule_path):
        return True
    return False