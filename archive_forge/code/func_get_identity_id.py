from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ssl import SSLError
def get_identity_id(rest_obj, module):
    """Get identity pool id based on requested identity pool name."""
    identity_name = module.params['identity_pool_name']
    resp = rest_obj.get_all_report_details(IDENTITY_URI)
    for each in resp['report_list']:
        if each['Name'] == identity_name:
            identity_id = each['Id']
            break
    else:
        module.fail_json(msg="Unable to complete the operation because the requested identity pool with name '{0}' is not present.".format(identity_name))
    return identity_id