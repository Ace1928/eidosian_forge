from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
def get_specific_baseline(module, baseline_name, resp_data):
    """Get specific baseline."""
    baseline = None
    for each in resp_data['value']:
        if each['Name'] == baseline_name:
            baseline = each
            break
    else:
        module.exit_json(msg="Unable to complete the operation because the requested baseline with name '{0}' does not exist.".format(baseline_name), baseline_info=[])
    return baseline