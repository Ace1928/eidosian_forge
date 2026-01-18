from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def host_fcwwpn_update(self):
    to_be_removed = ':'.join(list(set(self.existing_fcwwpn) - set(self.input_fcwwpn)))
    if to_be_removed:
        self.restapi.svc_run_command('rmhostport', {'fcwwpn': to_be_removed, 'force': True}, [self.name])
        self.log('%s removed from %s', to_be_removed, self.name)
    to_be_added = ':'.join(list(set(self.input_fcwwpn) - set(self.existing_fcwwpn)))
    if to_be_added:
        self.restapi.svc_run_command('addhostport', {'fcwwpn': to_be_added, 'force': True}, [self.name])
        self.log('%s added to %s', to_be_added, self.name)