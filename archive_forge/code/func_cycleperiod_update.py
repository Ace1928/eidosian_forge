from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils.basic import AnsibleModule
from traceback import format_exc
def cycleperiod_update(self):
    """
        Use the chrcrelationship command to update cycling period in remote copy
        relationship.
        """
    if self.module.check_mode:
        self.changed = True
        return
    if self.copytype == 'GMCV' and self.cyclingperiod:
        cmd = 'chrcrelationship'
        cmdopts = {}
        cmdopts['cycleperiodseconds'] = self.cyclingperiod
        cmdargs = [self.name]
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    else:
        self.log('not updating chrcrelationship with cyclingperiod %s', self.cyclingperiod)