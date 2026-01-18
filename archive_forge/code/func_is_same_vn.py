from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
import re
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
def is_same_vn(self, new, current):
    if new and current:
        return new == current
    else:
        return not current and (not new)