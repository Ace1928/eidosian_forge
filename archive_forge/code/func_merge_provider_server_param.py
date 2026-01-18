from __future__ import absolute_import, division, print_function
import copy
import os
import re
import datetime
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import exec_command
from ansible.module_utils.six import iteritems
from ansible.module_utils.parsing.convert_bool import (
from collections import defaultdict
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from .constants import (
def merge_provider_server_param(self, result, provider):
    if self.validate_params('server', provider):
        result['server'] = provider['server']
    elif self.validate_params('F5_SERVER', os.environ):
        result['server'] = os.environ['F5_SERVER']
    else:
        raise F5ModuleError('Server parameter cannot be None or missing, please provide a valid value')