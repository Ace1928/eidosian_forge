from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import remove_key
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination
def get_formatted_categories(rest_obj):
    report = get_all_data_with_pagination(rest_obj, ALERT_CATEGORY_URI)
    categories = remove_key(report.get('report_list', []))
    return categories