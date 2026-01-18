from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def run_luks_remove_key(self, device, keyfile, passphrase, keyslot, force_remove_last_key=False):
    """ Remove key from given device
            Raises ValueError when command fails
        """
    if not force_remove_last_key:
        result = self._run_command([self._cryptsetup_bin, 'luksDump', device])
        if result[RETURN_CODE] != 0:
            raise ValueError('Error while dumping LUKS header from %s' % (device,))
        keyslot_count = 0
        keyslot_area = False
        keyslot_re = re.compile('^Key Slot [0-9]+: ENABLED')
        for line in result[STDOUT].splitlines():
            if line.startswith('Keyslots:'):
                keyslot_area = True
            elif line.startswith('  '):
                if keyslot_area and line[2] in '0123456789':
                    keyslot_count += 1
            elif line.startswith('\t'):
                pass
            elif keyslot_re.match(line):
                keyslot_count += 1
            else:
                keyslot_area = False
        if keyslot_count < 2:
            self._module.fail_json(msg='LUKS device %s has less than two active keyslots. To be able to remove a key, please set `force_remove_last_key` to `true`.' % device)
    if keyslot is None:
        args = [self._cryptsetup_bin, 'luksRemoveKey', device, '-q']
    else:
        args = [self._cryptsetup_bin, 'luksKillSlot', device, '-q', str(keyslot)]
    if keyfile:
        args.extend(['--key-file', keyfile])
    result = self._run_command(args, data=passphrase)
    if result[RETURN_CODE] != 0:
        raise ValueError('Error while removing LUKS key from %s: %s' % (device, result[STDERR]))