from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def migrate_pools(self):
    self.basic_checks_migrate_vdisk()
    if self.module.check_mode:
        self.changed = True
        return
    source_data, target_data = self.get_existing_vdisk()
    if not source_data:
        msg = 'Source volume [%s] does not exist' % self.source_volume
        self.module.fail_json(msg=msg)
    elif source_data[0]['mdisk_grp_name'] != self.new_pool:
        cmd = 'migratevdisk'
        cmdopts = {}
        cmdopts['mdiskgrp'] = self.new_pool
        cmdopts['vdisk'] = self.source_volume
        self.log('Command %s opts %s', cmd, cmdopts)
        result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        if result == '':
            self.changed = True
        else:
            self.module.fail_json(msg='Failed to migrate volume in different pool.')
    else:
        msg = 'No modifications done. New pool [%s] is same' % self.new_pool
        self.module.exit_json(msg=msg, changed=False)