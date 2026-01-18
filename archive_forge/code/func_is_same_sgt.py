from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
import re
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
def is_same_sgt(self, new, current):

    def clean_excess(name):
        if name:
            return re.sub('\\s*\\(.*\\)$', '', name)
        else:
            return name
    has_new = self.get_sgt_by_id(new) or self.get_sgt_by_name(clean_excess(new))
    has_current = self.get_sgt_by_id(current) or self.get_sgt_by_name(clean_excess(current))
    if has_new and has_current:
        return has_new.get('id') == has_current.get('id')
    else:
        return not has_current and (not has_new)