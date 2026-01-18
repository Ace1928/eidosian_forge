from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def create_parameter_validation(self):
    if self.state == 'present':
        if not self.remote_clusterip:
            self.module.fail_json(msg='Missing required parameter during creation: remote_clusterip')
        if not (self.link1 or self.link2):
            self.module.fail_json(msg='At least one is required during creation: link1 or link2')
        if not (self.remote_link1 or self.remote_link2):
            self.module.fail_json(msg='At least one is required during creation: remote_link1 or remote_link2')