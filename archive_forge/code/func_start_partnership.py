from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def start_partnership(self, rest_object, id):
    cmd = 'chpartnership'
    cmd_opts = {'start': True}
    cmd_args = [id]
    rest_object.svc_run_command(cmd, cmd_opts, cmd_args)
    self.log('Started the partnership %s.', id)