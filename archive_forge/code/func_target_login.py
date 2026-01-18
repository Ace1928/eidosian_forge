from __future__ import absolute_import, division, print_function
import glob
import os
import re
import socket
import time
from ansible.module_utils.basic import AnsibleModule
def target_login(module, target, portal=None, port=None):
    node_auth = module.params['node_auth']
    node_user = module.params['node_user']
    node_pass = module.params['node_pass']
    node_user_in = module.params['node_user_in']
    node_pass_in = module.params['node_pass_in']
    if node_user:
        params = [('node.session.auth.authmethod', node_auth), ('node.session.auth.username', node_user), ('node.session.auth.password', node_pass)]
        for name, value in params:
            cmd = [iscsiadm_cmd, '--mode', 'node', '--targetname', target, '--op=update', '--name', name, '--value', value]
            module.run_command(cmd, check_rc=True)
    if node_user_in:
        params = [('node.session.auth.username_in', node_user_in), ('node.session.auth.password_in', node_pass_in)]
        for name, value in params:
            cmd = '%s --mode node --targetname %s --op=update --name %s --value %s' % (iscsiadm_cmd, target, name, value)
            module.run_command(cmd, check_rc=True)
    cmd = [iscsiadm_cmd, '--mode', 'node', '--targetname', target, '--login']
    if portal is not None and port is not None:
        cmd.append('--portal')
        cmd.append('%s:%s' % (portal, port))
    module.run_command(cmd, check_rc=True)