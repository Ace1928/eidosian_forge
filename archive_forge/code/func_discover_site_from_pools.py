from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def discover_site_from_pools(self):
    self.log('Entering function discover_site_from_pools')
    poolA_site = self.poolA_data['site_name']
    poolB_site = self.poolB_data['site_name']
    return (poolA_site, poolB_site)