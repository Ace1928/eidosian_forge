from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def is_sg_exists(self):
    merged_result = {}
    data = self.restapi.svc_obj_info(cmd='lssafeguardedschedule', cmdopts=None, cmdargs=[self.name])
    if isinstance(data, list):
        for d in data:
            merged_result.update(d)
    else:
        merged_result = data
    self.sg_policy_details = merged_result
    return merged_result