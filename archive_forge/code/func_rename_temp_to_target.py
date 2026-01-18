from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def rename_temp_to_target(self, temp_name):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'chvdisk'
    cmdopts = {}
    cmdopts['name'] = self.target
    self.log('Rename %s to %s', cmd, cmdopts)
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs=[temp_name])