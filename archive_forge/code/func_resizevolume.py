from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def resizevolume(self):
    if self.thin is not None or self.deduplicated is not None or self.rsize is not None or (self.grainsize is not None) or (self.compressed is not None) or (self.poolA is not None) or (self.poolB is not None) or (self.type is not None):
        self.module.fail_json(msg="Volume already exists, Parameter 'thin', 'deduplicated', 'rsize', 'grainsize', 'compressed' 'PoolA', 'PoolB' or 'type' cannot be passed while resizing the volume.")
    if self.module.check_mode:
        self.changed = True
        return
    cmd = ''
    cmdopts = {}
    if self.vdisk_type == 'local hyperswap' and self.expand_flag:
        cmd = 'expandvolume'
    elif self.vdisk_type == 'local hyperswap' and self.shrink_flag:
        self.module.fail_json(msg='Size of a HyperSwap Volume cannot be shrinked')
    elif self.vdisk_type == 'standard mirror' and self.expand_flag:
        cmd = 'expandvdisksize'
    elif self.vdisk_type == 'standard mirror' and self.shrink_flag:
        cmd = 'shrinkvdisksize'
    elif self.vdisk_type != 'standard mirror' or self.vdisk_type != 'local hyperswap':
        self.module.fail_json(msg='The volume is not a mirror volume, Please use ibm_svc_vdisk module for resizing standard volumes')
    cmdopts['size'] = str(self.changebysize)
    cmdopts['unit'] = 'b'
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs=[self.name])
    self.changed = True