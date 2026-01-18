from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
@cmd_runner_fmt.unpack_args
def execute_func(execute, manifest):
    if execute:
        return ['--execute', execute]
    else:
        return [manifest]