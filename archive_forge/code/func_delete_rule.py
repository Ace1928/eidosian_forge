from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
def delete_rule(self, rule_id):
    with self.plugin.ignore_not_found:
        self.client.delete_security_group_rule(rule_id)