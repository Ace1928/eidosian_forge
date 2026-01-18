from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def discover_system_topology(self):
    self.log('Entering function discover_system_topology')
    system_data = self.restapi.svc_obj_info(cmd='lssystem', cmdopts=None, cmdargs=None)
    sys_topology = system_data['topology']
    return sys_topology