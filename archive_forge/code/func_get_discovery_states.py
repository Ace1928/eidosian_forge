from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def get_discovery_states(rest_obj, key='JobStatusId'):
    resp = rest_obj.invoke_request('GET', DISCOVERY_JOBS_URI)
    disc_jobs = resp.json_data.get('value')
    job_state_dict = dict([(item['DiscoveryConfigGroupId'], item[key]) for item in disc_jobs])
    return job_state_dict