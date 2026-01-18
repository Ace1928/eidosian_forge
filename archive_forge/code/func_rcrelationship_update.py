from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils.basic import AnsibleModule
from traceback import format_exc
def rcrelationship_update(self, modify, modifycv):
    """
        Use the chrcrelationship command to modify certain attributes of an
        existing relationship, such as to add a relationship to a consistency
        group to remove a relationship from a consistency group.
        You can change one attribute at a time.
        """
    if self.module.check_mode:
        self.changed = True
        return
    if modify:
        self.log('updating chrcrelationship with properties %s', modify)
        cmd = 'chrcrelationship'
        cmdopts = {}
        for prop in modify:
            cmdopts[prop] = modify[prop]
        cmdargs = [self.name]
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.changed = True
    if modifycv:
        if 'cycleperiodseconds' in modifycv:
            self.cycleperiod_update()
            self.log('cyclingperiod in change volume updated')
        if 'cyclingmode' in modifycv:
            self.cyclemode_update()
            self.log('cyclingmode in change volume updated')
        self.changed = True
    if not modify and (not modifycv):
        self.log('There is no property need to be updated')
        self.changed = False