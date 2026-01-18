from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def lsvolumegroupsnapshot(self, force=False, old_name=None, parentuid=None):
    old_name = old_name if old_name else self.name
    if self.lsvg_data.get('name') == old_name and (not force):
        return self.lsvg_data
    cmdopts = {'snapshot': old_name}
    if parentuid:
        cmdopts['parentuid'] = self.parentuid
    else:
        cmdopts['volumegroup'] = self.volumegroup
    data = {}
    result = self.restapi.svc_obj_info(cmd='lsvolumegroupsnapshot', cmdopts=cmdopts, cmdargs=None)
    if isinstance(result, list):
        for res in result:
            data = res
    else:
        data = result
    self.lsvg_data = data
    return data