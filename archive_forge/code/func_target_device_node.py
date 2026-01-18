from __future__ import absolute_import, division, print_function
import glob
import os
import re
import socket
import time
from ansible.module_utils.basic import AnsibleModule
def target_device_node(target):
    devices = glob.glob('/dev/disk/by-path/*%s*' % target)
    devdisks = []
    for dev in devices:
        if '-part' not in dev:
            devdisk = os.path.realpath(dev)
            if devdisk not in devdisks:
                devdisks.append(devdisk)
    return devdisks