from __future__ import absolute_import, division, print_function
import crypt
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import jsonify
from ansible.module_utils.common.text.formatters import human_to_bytes
def homed_service_active(self):
    is_active = True
    cmd = ['systemctl', 'show', 'systemd-homed.service', '-p', 'ActiveState']
    rc, show_service_stdout, stderr = self.module.run_command(cmd)
    if rc == 0:
        state = show_service_stdout.rsplit('=')[1]
        if state.strip() != 'active':
            is_active = False
    return is_active