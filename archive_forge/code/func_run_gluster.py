from __future__ import absolute_import, division, print_function
import re
import socket
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def run_gluster(gargs, **kwargs):
    global glusterbin
    global module
    args = [glusterbin, '--mode=script']
    args.extend(gargs)
    try:
        rc, out, err = module.run_command(args, **kwargs)
        if rc != 0:
            module.fail_json(msg='error running gluster (%s) command (rc=%d): %s' % (' '.join(args), rc, out or err), exception=traceback.format_exc())
    except Exception as e:
        module.fail_json(msg='error running gluster (%s) command: %s' % (' '.join(args), to_native(e)), exception=traceback.format_exc())
    return out