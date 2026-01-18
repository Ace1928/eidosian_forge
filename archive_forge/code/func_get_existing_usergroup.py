from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def get_existing_usergroup(self):
    merged_result = {}
    data = self.restapi.svc_obj_info(cmd='lsusergrp', cmdopts=None, cmdargs=[self.name])
    self.log('GET: user group data: %s', data)
    if isinstance(data, list):
        for d in data:
            merged_result.update(d)
    else:
        merged_result = data
    return merged_result