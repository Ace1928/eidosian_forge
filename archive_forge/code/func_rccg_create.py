from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
def rccg_create(self):
    if self.module.check_mode:
        self.changed = True
        return
    rccg_data = self.get_existing_rccg()
    if rccg_data:
        self.rccg_update(rccg_data)
    self.log("creating rc consistgrp '%s'", self.name)
    cmd = 'mkrcconsistgrp'
    cmdopts = {'name': self.name}
    if self.cluster:
        cmdopts['cluster'] = self.cluster
    self.log("creating rc consistgrp command '%s' opts", self.cluster)
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log("create rc consistgrp result '%s'", result)
    msg = "succeeded to create rc consistgrp '%s'" % self.name
    self.log(msg)
    if 'message' in result:
        self.log("create rc consistgrp result message '%s'", result['message'])
        self.module.exit_json(msg="rc consistgrp '%s' is created" % self.name, changed=True)
    else:
        self.module.fail_json(msg=result)