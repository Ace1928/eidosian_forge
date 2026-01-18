from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_baseline_ids(rest_obj, module):
    """Getting the list of group ids filtered from the groups."""
    resp = rest_obj.get_all_report_details(BASELINE_URI)
    baseline, baseline_details = (module.params.get('baseline_name'), {})
    if resp['report_list']:
        for bse in resp['report_list']:
            if bse['Name'] == baseline:
                baseline_details['baseline_id'] = bse['Id']
                baseline_details['repo_id'] = bse['RepositoryId']
                baseline_details['catalog_id'] = bse['CatalogId']
        if not baseline_details:
            module.fail_json(msg="Unable to complete the operation because the entered target baseline name '{0}' is invalid.".format(baseline))
    else:
        module.fail_json(msg='Unable to complete the operation because the entered target baseline name does not exist.')
    return baseline_details