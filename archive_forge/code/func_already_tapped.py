from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def already_tapped(module, brew_path, tap):
    """Returns True if already tapped."""
    rc, out, err = module.run_command([brew_path, 'tap'])
    taps = [tap_.strip().lower() for tap_ in out.split('\n') if tap_]
    tap_name = re.sub('homebrew-', '', tap.lower())
    return tap_name in taps