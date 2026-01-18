from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
def convert_list_string(self, pList):
    if isinstance(pList, list):
        if len(pList) > 0:
            pList_str = list(map(str, pList))
            return ', '.join(pList_str)
        else:
            return ''
    else:
        return pList