from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def find_modify(self, current, desired_rules):
    if not current:
        return False
    if current.get('rules') is None or desired_rules is None:
        return False
    modify = False
    merge_rules = desired_rules['patch_rules'] + desired_rules['post_rules']
    if len(current.get('rules')) != len(merge_rules):
        return True
    for i in range(len(current['rules'])):
        if current['rules'][i]['index'] != merge_rules[i]['index'] or current['rules'][i]['type'] != merge_rules[i]['type']:
            return True
        else:
            if merge_rules[i].get('message_criteria') is None:
                merge_rules[i]['message_criteria'] = {'severities': '*', 'name_pattern': '*'}
            elif merge_rules[i]['message_criteria'].get('severities') is None:
                merge_rules[i]['message_criteria']['severities'] = '*'
            elif merge_rules[i]['message_criteria'].get('name_pattern') is None:
                merge_rules[i]['message_criteria']['name_pattern'] = '*'
            if current['rules'][i].get('message_criteria').get('name_pattern') != merge_rules[i].get('message_criteria').get('name_pattern'):
                return True
            if current['rules'][i].get('message_criteria').get('severities') != merge_rules[i].get('message_criteria').get('severities'):
                return True
    return modify