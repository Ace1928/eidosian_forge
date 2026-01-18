from __future__ import absolute_import, division, print_function
import os
import platform
import tempfile
from ansible.module_utils.basic import AnsibleModule
def get_matching_jobs(module, at_cmd, script_file):
    matching_jobs = []
    atq_cmd = module.get_bin_path('atq', True)
    atq_command = '%s' % atq_cmd
    rc, out, err = module.run_command(atq_command, check_rc=True)
    current_jobs = out.splitlines()
    if len(current_jobs) == 0:
        return matching_jobs
    with open(script_file) as script_fh:
        script_file_string = script_fh.read().strip()
    for current_job in current_jobs:
        split_current_job = current_job.split()
        at_opt = '-c' if platform.system() != 'AIX' else '-lv'
        at_command = '%s %s %s' % (at_cmd, at_opt, split_current_job[0])
        rc, out, err = module.run_command(at_command, check_rc=True)
        if script_file_string in out:
            matching_jobs.append(split_current_job[0])
    return matching_jobs