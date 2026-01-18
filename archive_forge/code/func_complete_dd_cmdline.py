from __future__ import absolute_import, division, print_function
import re
import os
import math
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def complete_dd_cmdline(args, dd_cmd):
    """Compute dd options to grow or truncate a file."""
    if args['file_size'] == args['size_spec']['bytes'] and (not args['force']):
        return list()
    bs = args['size_spec']['blocksize']
    if args['sparse']:
        seek = args['size_spec']['blocks']
    elif args['force'] or not os.path.exists(args['path']):
        seek = 0
    elif args['size_diff'] < 0:
        seek = args['size_spec']['blocks']
    elif args['size_diff'] % bs:
        seek = int(args['file_size'] / bs) + 1
    else:
        seek = int(args['file_size'] / bs)
    count = args['size_spec']['blocks'] - seek
    dd_cmd += ['bs=%s' % str(bs), 'seek=%s' % str(seek), 'count=%s' % str(count)]
    return dd_cmd