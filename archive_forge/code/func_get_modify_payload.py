from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def get_modify_payload(module, rest_obj, template_dict):
    modify_payload = {}
    attrib_dict = module.params.get('attributes')
    attrib_dict['Id'] = template_dict.get('Id')
    modify_payload['Name'] = template_dict['Name']
    diff = 0
    if attrib_dict.get('Name', template_dict['Name']) != template_dict['Name']:
        template = get_template_by_name(attrib_dict.get('Name'), module, rest_obj)
        if template:
            module.exit_json(msg=TEMPLATE_NAME_EXISTS.format(name=attrib_dict.get('Name')))
        modify_payload['Name'] = attrib_dict.get('Name')
        diff = diff + 1
    modify_payload['Description'] = template_dict['Description']
    diff = diff + apply_diff_key(attrib_dict, modify_payload, ['Description'])
    if attrib_dict.get('Attributes'):
        diff = diff + attributes_check(module, rest_obj, attrib_dict, template_dict.get('Id'))
    if not diff:
        module.exit_json(msg=NO_CHANGES_MSG)
    if isinstance(attrib_dict, dict):
        modify_payload.update(attrib_dict)
    return modify_payload