from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import raise_from
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def replace_rule(self):
    old_rule = self._get_rule()
    if old_rule:
        rules_order = self._get_rule_order()
        old_pos = rules_order.index(old_rule['id'])
        self.remove_role_perm()
        new_rule = self.create_role_perm()
        if new_rule:
            perm_order = self.order_permissions(int(old_pos - 1), new_rule['id'])
            return perm_order
    return None