from __future__ import absolute_import, division, print_function
import re
import socket
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def remove_bricks(name, removed_bricks, force):
    max_tries = 12
    retries = 0
    success = False
    args = ['volume', 'remove-brick', name]
    args.extend(removed_bricks)
    args_c = args[:]
    args.append('start')
    run_gluster(args)
    if not force:
        module.fail_json(msg='Force option is mandatory.')
    else:
        while retries < max_tries:
            last_brick = removed_bricks[-1]
            out = run_gluster(['volume', 'remove-brick', name, last_brick, 'status'])
            for row in out.split('\n')[1:]:
                if 'completed' in row:
                    args_c.append('commit')
                    out = run_gluster(args_c)
                    success = True
                    break
                else:
                    time.sleep(10)
            if success:
                break
            retries += 1
        if not success:
            module.fail_json(msg='Exceeded number of tries, check remove-brick status.\nCommit operation needs to be followed.')