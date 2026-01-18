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
def merge_provider_auth_provider_param(self, result, provider):
    if self.validate_params('auth_provider', provider):
        result['auth_provider'] = provider['auth_provider']
    elif self.validate_params('F5_AUTH_PROVIDER', os.environ):
        result['auth_provider'] = os.environ['F5_AUTH_PROVIDER']
    else:
        result['auth_provider'] = None
    if result['auth_provider'] is not None and '__omit_place_holder__' in result['auth_provider']:
        result['auth_provider'] = None