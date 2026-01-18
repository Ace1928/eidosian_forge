from __future__ import absolute_import, division, print_function
import copy
import os
import ssl
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.connection import exec_command
from ..module_utils.common import (
@property
def days_valid(self):
    if 1 <= self._values['days_valid'] <= 9125:
        return self._values['days_valid']
    raise F5ModuleError("Valid 'days_valid' must be in range 1 - 9125 days.")