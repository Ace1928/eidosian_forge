from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def run_luks_add_key(self, device, keyfile, passphrase, new_keyfile, new_passphrase, new_keyslot, pbkdf):
    """ Add new key from a keyfile or passphrase to given 'device';
            authentication done using 'keyfile' or 'passphrase'.
            Raises ValueError when command fails.
        """
    data = []
    args = [self._cryptsetup_bin, 'luksAddKey', device]
    if pbkdf is not None:
        self._add_pbkdf_options(args, pbkdf)
    if new_keyslot is not None:
        args.extend(['--key-slot', str(new_keyslot)])
    if keyfile:
        args.extend(['--key-file', keyfile])
    else:
        data.append(passphrase)
    if new_keyfile:
        args.append(new_keyfile)
    else:
        data.extend([new_passphrase, new_passphrase])
    result = self._run_command(args, data='\n'.join(data) or None)
    if result[RETURN_CODE] != 0:
        raise ValueError('Error while adding new LUKS keyslot to %s: %s' % (device, result[STDERR]))