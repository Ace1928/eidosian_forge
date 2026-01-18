from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def get_baseline_compliance_info(rest_obj, baseline_identifier_val, attribute='Id'):
    """
    Get the baseline info for the created compliance baseline
    """
    data = rest_obj.get_all_items_with_pagination(COMPLIANCE_BASELINE)
    value = data['value']
    baseline_info = {}
    for item in value:
        if item[attribute] == baseline_identifier_val:
            baseline_info = item
            baseline_info.pop('@odata.type', None)
            baseline_info.pop('@odata.id', None)
            baseline_info.pop('DeviceConfigComplianceReports@odata.navigationLink', None)
            break
    return baseline_info