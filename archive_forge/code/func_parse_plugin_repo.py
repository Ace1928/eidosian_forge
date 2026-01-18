from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def parse_plugin_repo(string):
    elements = string.split('/')
    repo = elements[0]
    if len(elements) > 1:
        repo = elements[1]
    for string in ('elasticsearch-', 'es-'):
        if repo.startswith(string):
            return repo[len(string):]
    return repo