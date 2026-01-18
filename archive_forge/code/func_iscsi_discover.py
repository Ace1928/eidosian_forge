from __future__ import absolute_import, division, print_function
import glob
import os
import re
import socket
import time
from ansible.module_utils.basic import AnsibleModule
def iscsi_discover(module, portal, port):
    cmd = [iscsiadm_cmd, '--mode', 'discovery', '--type', 'sendtargets', '--portal', '%s:%s' % (portal, port)]
    module.run_command(cmd, check_rc=True)