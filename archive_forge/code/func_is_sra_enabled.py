from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def is_sra_enabled(self):
    if self.sra_status_detail:
        return self.sra_status_detail['status'] == 'enabled'
    result = self.restapi.svc_obj_info(cmd='lssra', cmdopts=None, cmdargs=None)
    self.sra_status_detail = result
    return result['status'] == 'enabled'