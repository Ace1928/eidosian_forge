from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.storage_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def get_syslog_server_details(self, server_name):
    merged_result = {}
    data = self.restapi.svc_obj_info(cmd='lssyslogserver', cmdopts=None, cmdargs=[server_name])
    if isinstance(data, list):
        for d in data:
            merged_result.update(d)
    else:
        merged_result = data
    return merged_result