from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def get_job_data(discovery_json, rest_obj):
    job_list = discovery_json['DiscoveryConfigTaskParam']
    if len(job_list) == 1:
        job_id = job_list[0].get('TaskId')
    else:
        srch_key = 'DiscoveryConfigGroupId'
        srch_val = discovery_json[srch_key]
        resp = rest_obj.invoke_request('GET', DISCOVERY_JOBS_URI + '?$top=9999')
        discovs = resp.json_data.get('value')
        for d in discovs:
            if d[srch_key] == srch_val:
                job_id = d['JobId']
                break
    return job_id