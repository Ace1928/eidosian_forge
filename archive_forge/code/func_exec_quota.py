from __future__ import absolute_import, division, print_function
import grp
import os
import pwd
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
def exec_quota(module, xfs_quota_bin, cmd, mountpoint):
    cmd = [xfs_quota_bin, '-x', '-c'] + [cmd, mountpoint]
    rc, stdout, stderr = module.run_command(cmd, use_unsafe_shell=True)
    if 'XFS_GETQUOTA: Operation not permitted' in stderr.split('\n') or (rc == 1 and 'xfs_quota: cannot set limits: Operation not permitted' in stderr.split('\n')):
        module.fail_json(msg='You need to be root or have CAP_SYS_ADMIN capability to perform this operation')
    return (rc, stdout, stderr)