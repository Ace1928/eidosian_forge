from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict, job_tracking
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import CHANGES_MSG, NO_CHANGES_MSG
def formalize_job_payload(payload):
    payload['Id'] = 0
    payload['State'] = 'Enabled'
    prms = payload['Params']
    payload['Params'] = [{'Key': k, 'Value': v} for k, v in prms.items()]
    trgts = payload['Targets']
    payload['Targets'] = [{'Id': k, 'Data': '', 'TargetType': {'Id': 1000, 'Name': v}} for k, v in trgts.items()]
    jtype = payload['JobType']
    payload['JobType'] = {'Id': jtype, 'Name': jtype_map.get(jtype)}