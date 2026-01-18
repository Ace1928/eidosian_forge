from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import raise_from
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def order_permissions(self, parent, rule_id):
    rules = self._get_rule_order()
    if isinstance(parent, int):
        parent_pos = parent
    elif parent == '0':
        parent_pos = -1
    else:
        parent_rule = self._get_rule(parent)
        if not parent_rule:
            self.fail_json(msg="Parent rule '%s' not found" % parent)
        parent_pos = rules.index(parent_rule['id'])
    r_id = rules.pop(rules.index(rule_id))
    rules.insert(parent_pos + 1, r_id)
    rules = ','.join(map(str, rules))
    return rules