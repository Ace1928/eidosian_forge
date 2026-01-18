from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
def modify_profile(module, rest_obj):
    mparam = module.params
    payload = {}
    prof = get_profile(rest_obj, module)
    if not prof:
        module.fail_json(msg=PROFILE_NOT_FOUND.format(name=mparam.get('name')))
    diff = 0
    new_name = mparam.get('new_name')
    payload['Name'] = new_name if new_name else prof['ProfileName']
    if new_name and new_name != prof['ProfileName']:
        diff += 1
    desc = mparam.get('description')
    if desc and desc != prof['ProfileDescription']:
        payload['Description'] = desc
        diff += 1
    boot_iso_dict = get_network_iso_payload(module)
    rdict = prof.get('NetworkBootToIso') if prof.get('NetworkBootToIso') else {}
    if boot_iso_dict:
        nest_diff = recursive_diff(boot_iso_dict, rdict)
        if nest_diff:
            if nest_diff[0]:
                diff += 1
        payload['NetworkBootToIso'] = boot_iso_dict
    ad_opts = mparam.get('attributes')
    if ad_opts and ad_opts.get('Attributes'):
        diff = diff + attributes_check(module, rest_obj, ad_opts, prof['Id'])
        if ad_opts.get('Attributes'):
            payload['Attributes'] = ad_opts.get('Attributes')
    payload['Id'] = prof['Id']
    if diff:
        if module.check_mode:
            module.exit_json(msg=CHANGES_MSG, changed=True)
        rest_obj.invoke_request('PUT', PROFILE_VIEW + '({0})'.format(payload['Id']), data=payload)
        module.exit_json(msg='Successfully modified the profile.', changed=True)
    module.exit_json(msg=NO_CHANGES_MSG)