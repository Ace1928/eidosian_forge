from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
from tempfile import NamedTemporaryFile
from datetime import datetime
@rule_control.setter
def rule_control(self, control):
    if control.startswith('['):
        control = control.replace(' = ', '=').replace('[', '').replace(']', '')
        self._control = control.split(' ')
    else:
        self._control = control