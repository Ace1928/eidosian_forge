from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def isdrpool(self):
    poolA_drp = self.poolA_data['data_reduction']
    poolB_drp = self.poolB_data['data_reduction']
    isdrpool_list = [poolA_drp, poolB_drp]
    if 'yes' in isdrpool_list:
        self.isdrp = True