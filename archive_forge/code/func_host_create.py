from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def host_create(self):
    if not self.fcwwpn and (not self.iscsiname) and (not self.nqn):
        self.module.fail_json(msg='You must pass in fcwwpn or iscsiname or nqn to the module.')
    if self.fcwwpn and self.iscsiname or (self.nqn and self.iscsiname) or (self.fcwwpn and self.nqn):
        self.module.fail_json(msg='You have to pass only one parameter among fcwwpn, nqn and iscsiname to the module.')
    if self.hostcluster and self.nohostcluster:
        self.module.fail_json(msg='You must not pass in both hostcluster and nohostcluster to the module.')
    if self.module.check_mode:
        self.changed = True
        return
    self.log("creating host '%s'", self.name)
    cmd = 'mkhost'
    cmdopts = {'name': self.name, 'force': True}
    if self.fcwwpn:
        cmdopts['fcwwpn'] = self.fcwwpn
    elif self.iscsiname:
        cmdopts['iscsiname'] = self.iscsiname
    else:
        cmdopts['nqn'] = self.nqn
    cmdopts['protocol'] = self.protocol if self.protocol else 'scsi'
    if self.iogrp:
        cmdopts['iogrp'] = self.iogrp
    if self.type:
        cmdopts['type'] = self.type
    if self.site:
        cmdopts['site'] = self.site
    if self.portset:
        cmdopts['portset'] = self.portset
    self.log("creating host command '%s' opts '%s'", self.fcwwpn, self.type)
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log("create host result '%s'", result)
    if result and 'message' in result:
        self.changed = True
        self.log("create host result message '%s'", result['message'])
    else:
        self.module.fail_json(msg='Failed to create host [%s]' % self.name)