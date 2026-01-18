from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def target_create(self, temp_target_name, sdata):
    cmd = 'mkvdisk'
    cmdopts = {}
    cmdopts['name'] = temp_target_name
    if self.mdiskgrp:
        cmdopts['mdiskgrp'] = self.mdiskgrp
    else:
        cmdopts['mdiskgrp'] = sdata['mdisk_grp_name']
    cmdopts['size'] = sdata['capacity']
    cmdopts['unit'] = 'b'
    cmdopts['iogrp'] = sdata['IO_group_name']
    if self.copytype == 'snapshot':
        cmdopts['rsize'] = '0%'
        cmdopts['autoexpand'] = True
    if self.module.check_mode:
        self.changed = True
        return
    self.log('Creating vdisk.. Command %s opts %s', cmd, cmdopts)
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log('Create target volume result %s', result)
    if 'message' in result:
        self.changed = True
        self.log('Create target volume result message %s', result['message'])
    else:
        self.module.fail_json(msg='Failed to create target volume [%s]' % self.target)