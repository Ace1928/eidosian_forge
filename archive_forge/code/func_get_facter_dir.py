from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
def get_facter_dir():
    if os.getuid() == 0:
        return '/etc/facter/facts.d'
    else:
        return os.path.expanduser('~/.facter/facts.d')