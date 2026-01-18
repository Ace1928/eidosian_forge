from __future__ import absolute_import, division, print_function
import errno
import os
import platform
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.posix.plugins.module_utils.mount import ismount
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
def remount(module, args):
    """Try to use 'remount' first and fallback to (u)mount if unsupported."""
    mount_bin = module.get_bin_path('mount', required=True)
    cmd = [mount_bin]
    if platform.system().lower().endswith('bsd'):
        if module.params['state'] == 'remounted' and args['opts'] != 'defaults':
            cmd += ['-u', '-o', args['opts']]
        else:
            cmd += ['-u']
    elif module.params['state'] == 'remounted' and args['opts'] != 'defaults':
        cmd += ['-o', 'remount,' + args['opts']]
    else:
        cmd += ['-o', 'remount']
    if platform.system().lower() == 'openbsd':
        if module.params['fstab'] is not None:
            module.fail_json(msg='OpenBSD does not support alternate fstab files. Do not specify the fstab parameter for OpenBSD hosts')
    elif module.params['state'] != 'ephemeral':
        cmd += _set_fstab_args(args['fstab'])
    if module.params['state'] == 'ephemeral':
        cmd += _set_ephemeral_args(args)
    cmd += [args['name']]
    out = err = ''
    try:
        if module.params['state'] != 'ephemeral' and platform.system().lower().endswith('bsd'):
            rc = 1
        else:
            rc, out, err = module.run_command(cmd)
    except Exception:
        rc = 1
    msg = ''
    if rc != 0:
        msg = out + err
        if module.params['state'] == 'remounted' and args['opts'] != 'defaults':
            module.fail_json(msg='Options were specified with remounted, but the remount command failed. Failing in order to prevent an unexpected mount result. Try replacing this command with a "state: unmounted" followed by a "state: mounted" using the full desired mount options instead.')
        rc, msg = umount(module, args['name'])
        if rc == 0:
            rc, msg = mount(module, args)
    return (rc, msg)