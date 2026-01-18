from __future__ import absolute_import, division, print_function
import json
import os
import shutil
from ansible.module_utils.six import raise_from
def get_podman_version(module, fail=True):
    executable = module.params['executable'] if module.params['executable'] else 'podman'
    rc, out, err = module.run_command([executable, b'--version'])
    if rc != 0 or not out or 'version' not in out:
        if fail:
            module.fail_json(msg="'%s --version' run failed! Error: %s" % (executable, err))
        return None
    return out.split('version')[1].strip()