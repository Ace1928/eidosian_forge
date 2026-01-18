from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.storage_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def create_storage_partition(self):
    unsupported = ('noreplicationpolicy', 'preferredmanagementsystem', 'deletepreferredmanagementcopy')
    unsupported_exists = ', '.join((field for field in unsupported if getattr(self, field) not in {'', None}))
    if unsupported_exists:
        self.module.fail_json(msg='Paramters not supported while creation: {0}'.format(unsupported_exists))
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'mkpartition'
    cmdopts = {'name': self.name}
    if self.replicationpolicy:
        cmdopts['replicationpolicy'] = self.replicationpolicy
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log('Storage Partition (%s) created', self.name)
    self.changed = True