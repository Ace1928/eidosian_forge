from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def is_remote_support_enabled(self):
    if self.sra_status_detail:
        return self.sra_status_detail['remote_support_enabled'] == 'yes'
    result = self.restapi.svc_obj_info(cmd='lssra', cmdopts=None, cmdargs=None)
    return result['remote_support_enabled'] == 'yes'