from __future__ import absolute_import, division, print_function
import re
import os
import time
import tempfile
import filecmp
import shutil
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def initialize_from_null_state(initializer, initcommand, fallbackcmd, table):
    """
    This ensures iptables-state output is suitable for iptables-restore to roll
    back to it, i.e. iptables-save output is not empty. This also works for the
    iptables-nft-save alternative.
    """
    if table is None:
        table = 'filter'
    commandline = list(initializer)
    commandline += ['-t', table]
    dummy = module.run_command(commandline, check_rc=True)
    rc, out, err = module.run_command(initcommand, check_rc=True)
    if '*%s' % table not in out.splitlines():
        iptables_input = '*%s\n:OUTPUT ACCEPT\nCOMMIT\n' % table
        dummy = module.run_command(fallbackcmd, data=iptables_input, check_rc=True)
        rc, out, err = module.run_command(initcommand, check_rc=True)
    return (rc, out, err)