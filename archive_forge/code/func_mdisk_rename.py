from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.storage_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
def mdisk_rename(self, mdisk_data):
    msg = None
    old_mdisk_data = self.mdisk_exists(self.old_name)
    if not old_mdisk_data and (not mdisk_data):
        self.module.fail_json(msg='mdisk [{0}] does not exists.'.format(self.old_name))
    elif old_mdisk_data and mdisk_data:
        self.module.fail_json(msg='mdisk with name [{0}] already exists.'.format(self.name))
    elif not old_mdisk_data and mdisk_data:
        msg = 'mdisk [{0}] already renamed.'.format(self.name)
    elif old_mdisk_data and (not mdisk_data):
        if self.module.check_mode:
            self.changed = True
            return
        self.restapi.svc_run_command('chmdisk', {'name': self.name}, [self.old_name])
        self.changed = True
        msg = 'mdisk [{0}] has been successfully rename to [{1}].'.format(self.old_name, self.name)
    return msg