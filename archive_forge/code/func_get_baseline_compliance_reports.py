from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_baseline_compliance_reports(rest_obj, module):
    try:
        baseline_id = get_baseline_id_from_name(rest_obj, module)
        path = baselines_compliance_report_path.format(Id=baseline_id)
        resp_val = rest_obj.get_all_items_with_pagination(path)
        resp_data = resp_val['value']
        return resp_data
    except (URLError, HTTPError, SSLValidationError, ConnectionError, TypeError, ValueError) as err:
        raise err