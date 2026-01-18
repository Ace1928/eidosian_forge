from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def luks_test_key(self, device, keyfile, passphrase, keyslot=None):
    """ Check whether the keyfile or passphrase works.
            Raises ValueError when command fails.
        """
    data = None
    args = [self._cryptsetup_bin, 'luksOpen', '--test-passphrase', device]
    if keyfile:
        args.extend(['--key-file', keyfile])
    else:
        data = passphrase
    if keyslot is not None:
        args.extend(['--key-slot', str(keyslot)])
    result = self._run_command(args, data=data)
    if result[RETURN_CODE] == 0:
        return True
    for output in (STDOUT, STDERR):
        if 'No key available with this passphrase' in result[output]:
            return False
        if 'No usable keyslot is available.' in result[output]:
            return False
    if result[RETURN_CODE] == 1 and keyslot is not None and (result[STDOUT] == '') and (result[STDERR] == ''):
        return False
    raise ValueError('Error while testing whether keyslot exists on %s: %s' % (device, result[STDERR]))