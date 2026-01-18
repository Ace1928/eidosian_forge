from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def remove_proxy_details(self):
    if self.support == 'remote':
        cmd = 'rmsystemsupportcenter'
        cmdopts = {}
        for nm in self.name:
            if nm and nm != 'None':
                if self.is_proxy_exist(nm):
                    cmdargs = [nm]
                    self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
                    self.log('Proxy server(%s) details removed', nm)
                else:
                    self.log('Proxy server(%s) does not exist', nm)
            else:
                self.module.fail_json(msg='support is remote and state is disabled but following parameter is blank: name')