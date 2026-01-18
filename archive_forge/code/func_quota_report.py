from __future__ import absolute_import, division, print_function
import grp
import os
import pwd
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
def quota_report(module, xfs_quota_bin, mountpoint, name, quota_type, used_type):
    soft = None
    hard = None
    if quota_type == 'project':
        type_arg = '-p'
    elif quota_type == 'user':
        type_arg = '-u'
    elif quota_type == 'group':
        type_arg = '-g'
    if used_type == 'b':
        used_arg = '-b'
        used_name = 'blocks'
        factor = 1024
    elif used_type == 'i':
        used_arg = '-i'
        used_name = 'inodes'
        factor = 1
    elif used_type == 'rtb':
        used_arg = '-r'
        used_name = 'realtime blocks'
        factor = 1024
    rc, stdout, stderr = exec_quota(module, xfs_quota_bin, 'report %s %s' % (type_arg, used_arg), mountpoint)
    if rc != 0:
        result = dict(changed=False, rc=rc, stdout=stdout, stderr=stderr)
        module.fail_json(msg='Could not get quota report for %s.' % used_name, **result)
    for line in stdout.split('\n'):
        line = line.strip().split()
        if len(line) > 3 and line[0] == name:
            soft = int(line[2]) * factor
            hard = int(line[3]) * factor
            break
    return (soft, hard)