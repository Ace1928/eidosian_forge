from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def get_discovery_job(rest_obj, job_id):
    resp = rest_obj.invoke_request('GET', DISCOVERY_JOBS_URI + '({0})'.format(job_id))
    djob = resp.json_data
    nlist = list(djob)
    for k in nlist:
        if str(k).lower().startswith('@odata'):
            djob.pop(k)
    return djob