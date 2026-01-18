from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
def ensure_agent_enabled(module):
    runner = CmdRunner(module, command='puppet', path_prefix=_PUPPET_PATH_PREFIX, arg_formats=dict(_agent_disabled=cmd_runner_fmt.as_fixed(['config', 'print', 'agent_disabled_lockfile'])), check_rc=False)
    rc, stdout, stderr = runner('_agent_disabled').run()
    if os.path.exists(stdout.strip()):
        module.fail_json(msg='Puppet agent is administratively disabled.', disabled=True)
    elif rc != 0:
        module.fail_json(msg='Puppet agent state could not be determined.')