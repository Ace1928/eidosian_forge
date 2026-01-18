from __future__ import absolute_import, division, print_function
from collections import defaultdict
import re
from ansible.module_utils.basic import AnsibleModule
def run_pkgng(action, *args, **kwargs):
    cmd = [pkgng_path, dir_arg, action]
    pkgng_env = {'BATCH': 'yes'}
    if p['ignore_osver']:
        pkgng_env['IGNORE_OSVERSION'] = 'yes'
    if p['pkgsite'] is not None and action in ('update', 'install', 'upgrade'):
        if repo_flag_not_supported:
            pkgng_env['PACKAGESITE'] = p['pkgsite']
        else:
            cmd.append('--repository=%s' % (p['pkgsite'],))
    pkgng_env.update(kwargs.pop('environ_update', dict()))
    return module.run_command(cmd + list(args), environ_update=pkgng_env, **kwargs)