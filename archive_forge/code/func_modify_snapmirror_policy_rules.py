from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def modify_snapmirror_policy_rules(self, current=None, uuid=None):
    """
        Modify existing rules in snapmirror policy
        :return: None
        """
    if 'snapmirror_label' not in self.parameters:
        return
    obsolete_rules = self.identify_obsolete_snapmirror_policy_rules(current)
    new_rules = self.identify_new_snapmirror_policy_rules(current)
    modified_rules, unmodified_rules = self.identify_modified_snapmirror_policy_rules(current)
    self.rest_api.log_debug('OBS', obsolete_rules)
    self.rest_api.log_debug('NEW', new_rules)
    self.rest_api.log_debug('MOD', modified_rules)
    self.rest_api.log_debug('UNM', unmodified_rules)
    if self.use_rest:
        return self.modify_snapmirror_policy_rules_rest(uuid, obsolete_rules, unmodified_rules, modified_rules, new_rules)
    delete_rules = obsolete_rules + modified_rules
    add_schedule_rules, add_non_schedule_rules = self.identify_snapmirror_policy_rules_with_schedule(new_rules + modified_rules)
    for rule in delete_rules:
        options = {'policy-name': self.parameters['policy_name'], 'snapmirror-label': rule['snapmirror_label']}
        self.modify_snapmirror_policy_rule(options, 'snapmirror-policy-remove-rule')
    for rule in add_non_schedule_rules + add_schedule_rules:
        options = {'policy-name': self.parameters['policy_name'], 'snapmirror-label': rule['snapmirror_label'], 'keep': str(rule['keep'])}
        if 'prefix' in rule and rule['prefix'] != '':
            options['prefix'] = rule['prefix']
        if 'schedule' in rule and rule['schedule'] != '':
            options['schedule'] = rule['schedule']
        self.modify_snapmirror_policy_rule(options, 'snapmirror-policy-add-rule')