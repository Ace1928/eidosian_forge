from __future__ import absolute_import, division, print_function
import re
import socket
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def run_gluster_nofail(gargs, **kwargs):
    global glusterbin
    global module
    args = [glusterbin]
    args.extend(gargs)
    rc, out, err = module.run_command(args, **kwargs)
    if rc != 0:
        return None
    return out