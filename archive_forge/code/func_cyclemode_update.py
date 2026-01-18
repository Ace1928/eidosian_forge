from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils.basic import AnsibleModule
from traceback import format_exc
def cyclemode_update(self):
    """
        Use the chrcrelationship command to update cycling mode in remote copy
        relationship.
        """
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'chrcrelationship'
    cmdopts = {}
    cmdargs = [self.name]
    if self.copytype == 'GMCV':
        self.log('updating chrcrelationship with cyclingmode multi')
        cmdopts['cyclingmode'] = 'multi'
    else:
        self.log('updating chrcrelationship with no cyclingmode')
        cmdopts['cyclingmode'] = 'none'
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs)