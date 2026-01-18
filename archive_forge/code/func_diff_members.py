from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils import base
from ansible.module_utils.six.moves.urllib.parse import quote
def diff_members(target, current):
    diff = {'to_del': [], 'to_add': []}
    for member in target:
        if member not in current:
            diff['to_add'].append(member)
    for member in current:
        if member not in target:
            diff['to_del'].append(member)
    return diff