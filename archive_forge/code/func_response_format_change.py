from __future__ import (absolute_import, division, print_function)
import os
import json
from datetime import datetime
from os.path import exists
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.parse import urlparse
def response_format_change(response, params, file_name):
    resp = {}
    if params['job_wait']:
        if hasattr(response, 'json_data'):
            response = response.json_data
        response.pop('Description', None)
        response.pop('Name', None)
        response.pop('EndTime', None)
        response.pop('StartTime', None)
        response.pop('TaskState', None)
        response.pop('Messages', None)
        if response.get('Oem') is not None:
            response.update(response['Oem']['Dell'])
            response.pop('Oem', None)
        response = get_file(params, response, file_name)
        response['retval'] = True
    else:
        location = response.headers.get('Location')
        job_id = location.split('/')[-1]
        job_uri = JOB_URI.format(job_id=job_id)
        resp['Data'] = {'StatusCode': response.status_code, 'jobid': job_id, 'next_uri': job_uri}
        resp['Job'] = {'JobId': job_id, 'ResourceURI': job_uri}
        resp['Return'] = 'JobCreated'
        resp['Status'] = 'Success'
        resp['Message'] = 'none'
        resp['StatusCode'] = response.status_code
        resp = get_file(params, resp, file_name)
        resp['retval'] = True
        response = resp
    return response