from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def set_passphrase(module):
    """
    Attempt to set passphrase in the keyring using the Python API and fallback to using a shell.
    """
    if module.check_mode:
        return None
    try:
        keyring.set_password(module.params['service'], module.params['username'], module.params['user_password'])
        return None
    except keyring.errors.KeyringLocked:
        set_argument = 'echo "%s" | gnome-keyring-daemon --unlock\nkeyring set %s %s\n%s\n' % (quote(module.params['keyring_password']), quote(module.params['service']), quote(module.params['username']), quote(module.params['user_password']))
        dummy, dummy, stderr = module.run_command('dbus-run-session -- /bin/bash', use_unsafe_shell=True, data=set_argument, encoding=None)
        if not stderr.decode('UTF-8'):
            return None
        return stderr.decode('UTF-8')