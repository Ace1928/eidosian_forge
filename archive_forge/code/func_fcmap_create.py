from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def fcmap_create(self, temp_target_name):
    if self.copyrate:
        if self.copytype == 'clone':
            if int(self.copyrate) not in range(1, 151):
                self.module.fail_json(msg='Copyrate for clone must be in range 1-150')
        if self.copytype == 'snapshot':
            if int(self.copyrate) not in range(0, 151):
                self.module.fail_json(msg='Copyrate for snapshot must be in range 0-150')
    elif self.copytype == 'clone':
        self.copyrate = 50
    elif self.copytype == 'snapshot':
        self.copyrate = 0
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'mkfcmap'
    cmdopts = {}
    cmdopts['name'] = self.name
    cmdopts['source'] = self.source
    cmdopts['target'] = temp_target_name
    cmdopts['copyrate'] = self.copyrate
    if self.grainsize:
        cmdopts['grainsize'] = self.grainsize
    if self.consistgrp:
        cmdopts['consistgrp'] = self.consistgrp
    if self.copytype == 'clone':
        cmdopts['autodelete'] = True
    self.log('Creating fc mapping.. Command %s opts %s', cmd, cmdopts)
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log('Create flash copy mapping relationship result %s', result)
    if 'message' in result:
        self.changed = True
        self.log('Create flash copy mapping relationship result message %s', result['message'])
    else:
        self.module.fail_json(msg='Failed to create FlashCopy mapping relationship [%s]' % self.name)