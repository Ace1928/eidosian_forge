from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
@staticmethod
def route_to_string(route):
    result_str = ''
    result_str += route['ip']
    if route.get('next_hop') is not None:
        result_str += ' ' + route['next_hop']
    if route.get('metric') is not None:
        result_str += ' ' + str(route['metric'])
    for attribute, value in sorted(route.items()):
        if attribute not in ('ip', 'next_hop', 'metric') and value is not None:
            result_str += ' {0}={1}'.format(attribute, str(value).lower())
    return result_str