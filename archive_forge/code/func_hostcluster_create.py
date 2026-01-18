from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def hostcluster_create(self):
    if self.removeallhosts:
        self.module.fail_json(msg="Parameter 'removeallhosts' cannot be passed while creating hostcluster")
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'mkhostcluster'
    cmdopts = {'name': self.name}
    if self.ownershipgroup:
        cmdopts['ownershipgroup'] = self.ownershipgroup
    self.log("creating host cluster command opts '%s'", self.ownershipgroup)
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log("create host cluster result '%s'", result)
    if 'message' in result:
        self.changed = True
        self.log("create host cluster result message '%s'", result['message'])
    else:
        self.module.fail_json(msg='Failed to create host cluster [%s]' % self.name)