from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def del_passphrase(module):
    """
    Attempt to delete a passphrase in the keyring using the Python API and fallback to using a shell.
    """
    if module.check_mode:
        return None
    try:
        keyring.delete_password(module.params['service'], module.params['username'])
        return None
    except keyring.errors.KeyringLocked:
        delete_argument = 'echo "%s" | gnome-keyring-daemon --unlock\nkeyring del %s %s\n' % (quote(module.params['keyring_password']), quote(module.params['service']), quote(module.params['username']))
        dummy, dummy, stderr = module.run_command('dbus-run-session -- /bin/bash', use_unsafe_shell=True, data=delete_argument, encoding=None)
        if not stderr.decode('UTF-8'):
            return None
        return stderr.decode('UTF-8')