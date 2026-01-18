from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def addhostcluster(self):
    if self.module.check_mode:
        self.changed = True
        return
    self.log("Adding host '%s' in hostcluster %s", self.name, self.hostcluster)
    cmd = 'addhostclustermember'
    cmdopts = {}
    cmdargs = [self.hostcluster]
    cmdopts['host'] = self.name
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    self.changed = True