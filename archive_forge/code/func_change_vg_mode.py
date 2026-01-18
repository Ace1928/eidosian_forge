from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def change_vg_mode(self):
    cmd = 'chvolumegroupreplication'
    cmdopts = {}
    cmdopts['mode'] = self.mode
    self.log('Changing replicaiton direction.. Command %s opts %s', cmd, cmdopts)
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs=[self.name])