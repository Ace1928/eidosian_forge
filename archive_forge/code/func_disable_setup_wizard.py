from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import svc_ssh_argument_spec, get_logger
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_ssh import IBMSVCssh
def disable_setup_wizard(self):
    self.log('Disable setup wizard')
    cmd = 'chsystem -easysetup no'
    stdin, stdout, stderr = self.ssh_client.client.exec_command(cmd)