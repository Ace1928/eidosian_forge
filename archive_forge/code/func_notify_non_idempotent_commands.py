from __future__ import absolute_import, division, print_function
import copy
import re
import shlex
import time
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.parsing import Conditional
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from collections import deque
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def notify_non_idempotent_commands(self, commands):
    for index, item in enumerate(commands):
        if any((item.startswith(x) for x in self.valid_configs)):
            return
        else:
            self.warnings.append('Using "write" commands is not idempotent. You should use a module that is specifically made for that. If such a module does not exist, then please file a bug. The command in question is "{0}..."'.format(item[0:40]))