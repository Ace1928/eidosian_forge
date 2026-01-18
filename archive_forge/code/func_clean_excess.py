from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
import re
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
def clean_excess(name):
    if name:
        return re.sub('\\s*\\(.*\\)$', '', name)
    else:
        return name