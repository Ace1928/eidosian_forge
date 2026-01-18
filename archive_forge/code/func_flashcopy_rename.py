from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.storage_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def flashcopy_rename(self):
    msg = None
    self.parameter_handling_while_renaming()
    mdata = self.mdata_exists(self.name)
    old_mdata = self.mdata_exists(self.old_name)
    if not old_mdata and (not mdata):
        self.module.fail_json(msg='mapping [{0}] does not exists.'.format(self.old_name))
    elif old_mdata and mdata:
        self.module.fail_json(msg='mapping with name [{0}] already exists.'.format(self.name))
    elif not old_mdata and mdata:
        msg = 'mdisk [{0}] already renamed.'.format(self.name)
    elif old_mdata and (not mdata):
        if self.module.check_mode:
            self.changed = True
            return
        self.restapi.svc_run_command('chfcmap', {'name': self.name}, [self.old_name])
        self.changed = True
        msg = 'mapping [{0}] has been successfully rename to [{1}].'.format(self.old_name, self.name)
    return msg