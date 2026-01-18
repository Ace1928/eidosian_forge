from __future__ import absolute_import, division, print_function
import glob
import os
import re
import socket
import time
from ansible.module_utils.basic import AnsibleModule
def target_logout(module, target):
    cmd = [iscsiadm_cmd, '--mode', 'node', '--targetname', target, '--logout']
    module.run_command(cmd, check_rc=True)