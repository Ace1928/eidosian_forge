from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def is_portset_exists(self, portset_name):
    merged_result = {}
    data = self.restapi.svc_obj_info(cmd='lsportset', cmdopts=None, cmdargs=[portset_name])
    if isinstance(data, list):
        for d in data:
            merged_result.update(d)
    else:
        merged_result = data
    self.portset_details = merged_result
    return merged_result