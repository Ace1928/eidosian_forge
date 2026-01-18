import configparser
import re
from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import policy
import glance.api.policy
from glance.common import exception
from glance.i18n import _, _LE, _LW
def check_property_rules(self, property_name, action, context):
    roles = context.roles
    if context.service_roles:
        roles.extend(context.service_roles)
    if not self.rules:
        return True
    if action not in ['create', 'read', 'update', 'delete']:
        return False
    for rule_exp, rule in self.rules:
        if rule_exp.search(str(property_name)):
            break
    else:
        return False
    rule_roles = rule.get(action)
    if rule_roles:
        if '!' in rule_roles:
            return False
        elif '@' in rule_roles:
            return True
        if self.prop_prot_rule_format == 'policies':
            prop_exp_key = self.prop_exp_mapping[rule_exp]
            return self._check_policy(prop_exp_key, action, context)
        if set(roles).intersection(set([role.lower() for role in rule_roles])):
            return True
    return False