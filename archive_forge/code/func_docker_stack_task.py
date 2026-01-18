from __future__ import (absolute_import, division, print_function)
import json
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
def docker_stack_task(module, stack_name):
    docker_bin = module.get_bin_path('docker', required=True)
    rc, out, err = module.run_command([docker_bin, 'stack', 'ps', stack_name, '--format={{json .}}'])
    return (rc, out.strip(), err.strip())