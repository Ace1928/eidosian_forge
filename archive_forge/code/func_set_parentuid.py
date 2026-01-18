from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
def set_parentuid(self):
    if self.snapshot and (not self.fromsourcegroup):
        cmdopts = {'filtervalue': 'snapshot_name={0}'.format(self.snapshot)}
        data = self.restapi.svc_obj_info(cmd='lsvolumesnapshot', cmdopts=cmdopts, cmdargs=None)
        try:
            result = next(filter(lambda obj: obj['volume_group_name'] == '', data))
        except StopIteration:
            self.module.fail_json(msg='Orphan Snapshot ({0}) does not exists for the given name'.format(self.snapshot))
        else:
            self.parentuid = result['parent_uid']