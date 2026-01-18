from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def create_partnership(self, location, cluster_ip):
    if self.module.check_mode:
        self.changed = True
        return
    rest_api = None
    cmd = 'mkippartnership'
    cmd_opts = {'clusterip': cluster_ip}
    if self.type:
        cmd_opts['type'] = self.type
    if self.compressed:
        cmd_opts['compressed'] = self.compressed
    if self.linkbandwidthmbits:
        cmd_opts['linkbandwidthmbits'] = self.linkbandwidthmbits
    if self.backgroundcopyrate:
        cmd_opts['backgroundcopyrate'] = self.backgroundcopyrate
    if location == 'local':
        rest_api = self.restapi_local
        if self.link1:
            cmd_opts['link1'] = self.link1
        if self.link2:
            cmd_opts['link2'] = self.link2
    if location == 'remote':
        rest_api = self.restapi_remote
        if self.remote_link1:
            cmd_opts['link1'] = self.remote_link1
        if self.remote_link2:
            cmd_opts['link2'] = self.remote_link2
    result = rest_api.svc_run_command(cmd, cmd_opts, cmdargs=None)
    self.log("Create result '%s'.", result)
    if result == '':
        self.changed = True
        self.log('Created IP partnership for %s system.', location)
    else:
        self.module.fail_json(msg='Failed to create IP partnership for cluster ip {0}'.format(cluster_ip))