from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def desired_ems_rules(self, current_rules):
    current_rules['rules'] = current_rules['rules'][:-1]
    if self.parameters.get('rules'):
        input_rules = self.na_helper.filter_out_none_entries(self.parameters['rules'])
        for i in range(len(input_rules)):
            input_rules[i]['message_criteria']['severities'] = input_rules[i]['message_criteria']['severities'].lower()
        matched_idx = []
        patch_rules = []
        post_rules = []
        for rule_dict in current_rules['rules']:
            for i in range(len(input_rules)):
                if input_rules[i]['index'] == rule_dict['index']:
                    matched_idx.append(int(input_rules[i]['index']))
                    patch_rules.append(input_rules[i])
                    break
            else:
                rule = {'index': rule_dict['index']}
                rule['type'] = rule_dict.get('type')
                if 'message_criteria' in rule_dict:
                    rule['message_criteria'] = {}
                    rule['message_criteria']['severities'] = rule_dict.get('message_criteria').get('severities')
                    rule['message_criteria']['name_pattern'] = rule_dict.get('message_criteria').get('name_pattern')
                patch_rules.append(rule)
        for i in range(len(input_rules)):
            if int(input_rules[i]['index']) not in matched_idx:
                post_rules.append(input_rules[i])
        desired_rules = {'patch_rules': patch_rules, 'post_rules': post_rules}
        return desired_rules
    return None