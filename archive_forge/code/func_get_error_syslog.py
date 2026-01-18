from __future__ import (absolute_import, division, print_function)
import os
import json
import time
from ssl import SSLError
from xml.etree import ElementTree as ET
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_error_syslog(idrac, curr_time, uri):
    error_log_found = False
    msg = None
    error_log_ids = ['SYS229', 'SYS227', 'RED132', 'JCP042', 'RED068', 'RED137']
    intrvl = 5
    retries = 60 // intrvl
    try:
        if not curr_time:
            resp = idrac.invoke_request(LOG_SERVICE_URI, 'GET')
            uri = resp.json_data.get('Entries').get('@odata.id')
            curr_time = resp.json_data.get('DateTime')
        fltr = "?$filter=Created%20ge%20'{0}'".format(curr_time)
        fltr_uri = '{0}{1}'.format(uri, fltr)
        while retries:
            resp = idrac.invoke_request(fltr_uri, 'GET')
            logs_list = resp.json_data.get('Members')
            for log in logs_list:
                for err_id in error_log_ids:
                    if err_id in log.get('MessageId'):
                        error_log_found = True
                        msg = log.get('Message')
                        break
                if msg or error_log_found:
                    break
            if msg or error_log_found:
                break
            retries = retries - 1
            time.sleep(intrvl)
        else:
            msg = 'No Error log found.'
            error_log_found = False
    except Exception:
        msg = 'No Error log found.'
        error_log_found = False
    return (error_log_found, msg)