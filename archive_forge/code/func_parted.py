from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import math
import re
import os
def parted(script, device, align):
    """
    Runs a parted script.
    """
    global module, parted_exec
    align_option = '-a %s' % align
    if align == 'undefined':
        align_option = ''
    "\n    Use option --fix (-f) if available. Versions prior\n    to 3.4.64 don't have it. For more information see:\n    http://savannah.gnu.org/news/?id=10114\n    "
    if parted_version() >= (3, 4, 64):
        script_option = '-s -f'
    else:
        script_option = '-s'
    if script and (not module.check_mode):
        command = '%s %s -m %s %s -- %s' % (parted_exec, script_option, align_option, device, script)
        rc, out, err = module.run_command(command)
        if rc != 0:
            module.fail_json(msg='Error while running parted script: %s' % command.strip(), rc=rc, out=out, err=err)