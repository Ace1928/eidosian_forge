from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.storage_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def rename_server(self, server_data):
    msg = ''
    old_name_data = self.get_syslog_server_details(self.old_name)
    if not old_name_data and (not server_data):
        self.module.fail_json(msg="Syslog server with old name {0} doesn't exist.".format(self.old_name))
    elif old_name_data and server_data:
        self.module.fail_json(msg='Syslog server [{0}] already exists.'.format(self.name))
    elif not old_name_data and server_data:
        msg = 'Syslog server with name [{0}] already exists.'.format(self.name)
    elif old_name_data and (not server_data):
        if self.module.check_mode:
            self.changed = True
            return
        self.restapi.svc_run_command('chsyslogserver', {'name': self.name}, [self.old_name])
        self.changed = True
        msg = 'Syslog server [{0}] has been successfully rename to [{1}].'.format(self.old_name, self.name)
    return msg