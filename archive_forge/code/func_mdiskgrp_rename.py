from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
def mdiskgrp_rename(self, mdiskgrp_data):
    msg = None
    old_mdiskgrp_data = self.mdiskgrp_exists(self.old_name)
    if not old_mdiskgrp_data and (not mdiskgrp_data):
        self.module.fail_json(msg='mdiskgrp [{0}] does not exists.'.format(self.old_name))
    elif old_mdiskgrp_data and mdiskgrp_data:
        self.module.fail_json(msg='mdiskgrp with name [{0}] already exists.'.format(self.name))
    elif not old_mdiskgrp_data and mdiskgrp_data:
        msg = 'mdiskgrp [{0}] already renamed.'.format(self.name)
    elif old_mdiskgrp_data and (not mdiskgrp_data):
        if self.old_name == self.parentmdiskgrp:
            self.module.fail_json("Old name shouldn't be same as parentmdiskgrp while renaming childmdiskgrp")
        if self.module.check_mode:
            self.changed = True
            return
        self.restapi.svc_run_command('chmdiskgrp', {'name': self.name}, [self.old_name])
        self.changed = True
        msg = 'mdiskgrp [{0}] has been successfully rename to [{1}].'.format(self.old_name, self.name)
    return msg