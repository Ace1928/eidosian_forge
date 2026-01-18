from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def test_connection_cloud_callhome(self):
    if self.module.check_mode:
        self.changed = True
        return
    command = 'sendcloudcallhome'
    command_options = {'connectiontest': True}
    self.restapi.svc_run_command(command, command_options, None)
    self.changed = True
    self.log('Cloud callhome connection tested.')
    time.sleep(3)