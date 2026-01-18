from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def add_tap(module, brew_path, tap, url=None):
    """Adds a single tap."""
    failed, changed, msg = (False, False, '')
    if not a_valid_tap(tap):
        failed = True
        msg = 'not a valid tap: %s' % tap
    elif not already_tapped(module, brew_path, tap):
        if module.check_mode:
            module.exit_json(changed=True)
        rc, out, err = module.run_command([brew_path, 'tap', tap, url])
        if rc == 0:
            changed = True
            msg = 'successfully tapped: %s' % tap
        else:
            failed = True
            msg = 'failed to tap: %s due to %s' % (tap, err)
    else:
        msg = 'already tapped: %s' % tap
    return (failed, changed, msg)