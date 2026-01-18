from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def apt_repo(module, *args):
    """run apt-repo with args and return its output"""
    args = list(args)
    rc, out, err = module.run_command([APT_REPO_PATH] + args)
    if rc != 0:
        module.fail_json(msg="'%s' failed: %s" % (' '.join(['apt-repo'] + args), err))
    return out