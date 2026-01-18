from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def check_source(self):
    result = {}
    if self.volumegroup_name:
        cmd = 'lsvolumegroup'
        cmdargs = [self.volumegroup_name]
        cmdopts = None
    elif self.volume_name and self.state == 'present':
        cmd = 'lsvdisk'
        cmdargs = [self.volume_name]
        cmdopts = None
    else:
        cmd = 'lsvolumebackupgeneration'
        cmdargs = None
        cmdopts = {}
        if self.volume_UID:
            self.var = self.volume_UID
            cmdopts['uid'] = self.volume_UID
        else:
            self.var = self.volume_name
            cmdopts['volume'] = self.volume_name
    data = self.restapi.svc_obj_info(cmd=cmd, cmdopts=cmdopts, cmdargs=cmdargs)
    if isinstance(data, list):
        for d in data:
            result.update(d)
    else:
        result = data
    if self.state == 'present':
        return not result
    else:
        return result