from __future__ import absolute_import, division, print_function
import datetime
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import open_url
def get_component_ids(auth_content, components):
    host_ids = []
    lower_components = [x.lower() for x in components]
    for result in auth_content['result']:
        if result['name'].lower() in lower_components:
            data = {'component_id': result['_id'], 'container_id': result['containers'][0]['_id']}
            host_ids.append(data)
            lower_components.remove(result['name'].lower())
    if len(lower_components):
        return (1, None, lower_components)
    return (0, host_ids, None)