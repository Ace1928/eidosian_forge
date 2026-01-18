from __future__ import (absolute_import, division, print_function)
import json
import time
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import remove_key
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def modify_catalog(module, rest_obj, catalog_list, all_catalog):
    params = module.params
    catalog_id = catalog_list[0]['Id']
    name = catalog_list[0]['Repository']['Name']
    modify_payload = _get_catalog_payload(module.params, name)
    new_catalog_name = params.get('new_catalog_name')
    if new_catalog_name:
        if new_catalog_name != name and new_catalog_name in all_catalog:
            module.fail_json(msg=CATALOG_EXISTS.format(new_name=new_catalog_name))
        modify_payload['Repository']['Name'] = new_catalog_name
    catalog_payload = get_current_catalog_settings(catalog_list[0])
    if modify_payload.get('Repository') and modify_payload.get('Repository').get('RepositoryType') and (modify_payload.get('Repository').get('RepositoryType') != catalog_payload['Repository']['RepositoryType']):
        module.fail_json(msg='Repository type cannot be changed to another repository type.')
    new_catalog_current_setting = catalog_payload.copy()
    repo_id = new_catalog_current_setting['Repository']['Id']
    del new_catalog_current_setting['Repository']['Id']
    fname = modify_payload.get('Filename')
    if fname and fname.lower().endswith('.gz'):
        modify_payload['Filename'] = new_catalog_current_setting.get('Filename')
        src_path = modify_payload.get('SourcePath')
        if src_path is None:
            src_path = new_catalog_current_setting.get('SourcePath', '')
            if src_path.lower().endswith('.gz'):
                src_path = os.path.dirname(src_path)
        modify_payload['SourcePath'] = os.path.join(src_path, fname)
    diff = compare_payloads(modify_payload, new_catalog_current_setting)
    if not diff:
        module.exit_json(msg=CHECK_MODE_CHANGE_NOT_FOUND_MSG, changed=False)
    if module.check_mode:
        module.exit_json(msg=CHECK_MODE_CHANGE_FOUND_MSG, changed=True)
    new_catalog_current_setting['Repository'].update(modify_payload['Repository'])
    catalog_payload.update(modify_payload)
    catalog_payload['Repository'] = new_catalog_current_setting['Repository']
    catalog_payload['Repository']['Id'] = repo_id
    catalog_payload['Id'] = catalog_id
    catalog_put_uri = CATALOG_URI_ID.format(Id=catalog_id)
    resp = rest_obj.invoke_request('PUT', catalog_put_uri, data=catalog_payload)
    resp_data = resp.json_data
    job_id = resp_data.get('TaskId')
    msg = 'Successfully triggered the job to update a catalog with Task Id : {0}'.format(job_id)
    exit_catalog(module, rest_obj, resp_data, 'modified', msg)