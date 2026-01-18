from __future__ import absolute_import, division, print_function
import grp
import os
import pwd
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
def get_fs_by_mountpoint(mountpoint):
    mpr = None
    with open('/proc/mounts', 'r') as s:
        for line in s.readlines():
            mp = line.strip().split()
            if len(mp) == 6 and mp[1] == mountpoint and (mp[2] == 'xfs'):
                mpr = dict(zip(['spec', 'file', 'vfstype', 'mntopts', 'freq', 'passno'], mp))
                mpr['mntopts'] = mpr['mntopts'].split(',')
                break
    return mpr