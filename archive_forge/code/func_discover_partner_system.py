from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def discover_partner_system(self):
    cmd = 'lspartnership'
    cmdopts = {}
    cmdargs = [self.remote_cluster]
    partnership_data = self.restapi.svc_obj_info(cmd, cmdopts, cmdargs)
    if partnership_data:
        system_location = partnership_data['location']
        if system_location == 'local':
            self.module.fail_json(msg='The relationship could not be created as migration relationships are only allowed to be created to a remote system.')
        self.partnership_exists = True
        remote_socket = partnership_data['console_IP']
        return remote_socket.split(':')[0]
    else:
        msg = 'The partnership with remote cluster [%s] does not exist.' % self.remote_cluster
        self.module.fail_json(msg=msg)