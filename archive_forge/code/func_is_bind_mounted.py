from __future__ import absolute_import, division, print_function
import errno
import os
import platform
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.posix.plugins.module_utils.mount import ismount
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
def is_bind_mounted(module, linux_mounts, dest, src=None, fstype=None):
    """Return whether the dest is bind mounted

    :arg module: The AnsibleModule (used for helper functions)
    :arg dest: The directory to be mounted under. This is the primary means
        of identifying whether the destination is mounted.
    :kwarg src: The source directory. If specified, this is used to help
        ensure that we are detecting that the correct source is mounted there.
    :kwarg fstype: The filesystem type. If specified this is also used to
        help ensure that we are detecting the right mount.
    :kwarg linux_mounts: Cached list of mounts for Linux.
    :returns: True if the dest is mounted with src otherwise False.
    """
    is_mounted = False
    if platform.system() == 'Linux' and linux_mounts is not None:
        if src is None:
            if dest in linux_mounts:
                is_mounted = True
        elif dest in linux_mounts:
            is_mounted = linux_mounts[dest]['src'] == src
    else:
        bin_path = module.get_bin_path('mount', required=True)
        cmd = '%s -l' % bin_path
        rc, out, err = module.run_command(cmd)
        mounts = []
        if len(out):
            mounts = to_native(out).strip().split('\n')
        for mnt in mounts:
            arguments = mnt.split()
            if (arguments[0] == src or src is None) and arguments[2] == dest and (arguments[4] == fstype or fstype is None):
                is_mounted = True
            if is_mounted:
                break
    return is_mounted