from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def check_email_server_exists(self):
    status = False
    data = self.restapi.svc_obj_info(cmd='lsemailserver', cmdopts=None, cmdargs=None)
    for item in data:
        if item['IP_address'] == self.serverIP and int(item['port']) == self.serverPort:
            status = True
            break
    return status