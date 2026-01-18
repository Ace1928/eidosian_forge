from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import raise_from
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleError
from datetime import datetime
def process_integer(self, field_name, rule, min_value, max_value, rule_number):
    return_values = []
    if isinstance(rule[field_name], int):
        rule[field_name] = [rule[field_name]]
    if not isinstance(rule[field_name], list):
        rule[field_name] = rule[field_name].split(',')
    for value in rule[field_name]:
        if isinstance(value, str):
            value = value.strip()
        if not re.match('^\\d+$', str(value)) or int(value) < min_value or int(value) > max_value:
            raise AnsibleError('In rule {0} {1} must be between {2} and {3}'.format(rule_number, field_name, min_value, max_value))
        return_values.append(int(value))
    return return_values