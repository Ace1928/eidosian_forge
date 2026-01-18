from __future__ import (absolute_import, division, print_function)
import os
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector
def is_chroot(module=None):
    is_chroot = None
    if os.environ.get('debian_chroot', False):
        is_chroot = True
    else:
        my_root = os.stat('/')
        try:
            proc_root = os.stat('/proc/1/root/.')
            is_chroot = my_root.st_ino != proc_root.st_ino or my_root.st_dev != proc_root.st_dev
        except Exception:
            fs_root_ino = 2
            if module is not None:
                stat_path = module.get_bin_path('stat')
                if stat_path:
                    cmd = [stat_path, '-f', '--format=%T', '/']
                    rc, out, err = module.run_command(cmd)
                    if 'btrfs' in out:
                        fs_root_ino = 256
                    elif 'xfs' in out:
                        fs_root_ino = 128
            is_chroot = my_root.st_ino != fs_root_ino
    return is_chroot