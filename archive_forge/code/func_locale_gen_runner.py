from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
def locale_gen_runner(module):
    runner = CmdRunner(module, command='locale-gen', arg_formats=dict(name=cmd_runner_fmt.as_list(), purge=cmd_runner_fmt.as_fixed('--purge')), check_rc=True)
    return runner