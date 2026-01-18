from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def get_existing_cloud_callhome_data(self):
    data = {}
    command = 'lscloudcallhome'
    command_options = None
    cmdargs = None
    data = self.restapi.svc_obj_info(command, command_options, cmdargs)
    return data