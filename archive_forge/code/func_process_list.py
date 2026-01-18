from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import raise_from
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleError
from datetime import datetime
def process_list(self, field_name, rule, valid_list, rule_number):
    return_values = []
    if not isinstance(rule[field_name], list):
        rule[field_name] = rule[field_name].split(',')
    for value in rule[field_name]:
        value = value.strip().lower()
        if value not in valid_list:
            raise AnsibleError('In rule {0} {1} must only contain values in {2}'.format(rule_number, field_name, ', '.join(valid_list.keys())))
        return_values.append(valid_list[value])
    return return_values