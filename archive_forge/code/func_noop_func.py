from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
def noop_func(v):
    return ['--noop'] if module.check_mode or v else ['--no-noop']