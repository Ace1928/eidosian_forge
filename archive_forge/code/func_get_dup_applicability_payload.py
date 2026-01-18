from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_dup_applicability_payload(file_token, device_ids=None, group_ids=None, baseline_ids=None):
    """Returns the DUP applicability JSON payload."""
    dup_applicability_payload = {'SingleUpdateReportBaseline': [], 'SingleUpdateReportGroup': [], 'SingleUpdateReportTargets': [], 'SingleUpdateReportFileToken': file_token}
    if device_ids is not None:
        dup_applicability_payload.update({'SingleUpdateReportTargets': list(map(int, device_ids))})
    elif group_ids is not None:
        dup_applicability_payload.update({'SingleUpdateReportGroup': list(map(int, group_ids))})
    elif baseline_ids is not None:
        dup_applicability_payload.update({'SingleUpdateReportBaseline': list(map(int, baseline_ids))})
    return dup_applicability_payload