from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def create_rule_spec(self):
    """
        Create DRS rule
        """
    changed = False
    result = None
    if self.affinity_rule:
        rule = vim.cluster.AffinityRuleSpec()
    else:
        rule = vim.cluster.AntiAffinityRuleSpec()
    rule.vm = self.vm_obj_list
    rule.enabled = self.enabled
    rule.mandatory = self.mandatory
    rule.name = self.rule_name
    rule_spec = vim.cluster.RuleSpec(info=rule, operation='add')
    config_spec = vim.cluster.ConfigSpecEx(rulesSpec=[rule_spec])
    try:
        if not self.module.check_mode:
            task = self.cluster_obj.ReconfigureEx(config_spec, modify=True)
            changed, result = wait_for_task(task)
    except vmodl.fault.InvalidRequest as e:
        result = to_native(e.msg)
    except Exception as e:
        result = to_native(e)
    if changed:
        rule_obj = self.get_rule_key_by_name(rule_name=self.rule_name)
        result = self.normalize_rule_spec(rule_obj)
    if self.module.check_mode:
        changed = True
        result = dict(rule_key='', rule_enabled=rule.enabled, rule_name=self.rule_name, rule_mandatory=rule.mandatory, rule_uuid='', rule_vms=[vm.name for vm in rule.vm], rule_affinity=self.affinity_rule)
    return (changed, result)