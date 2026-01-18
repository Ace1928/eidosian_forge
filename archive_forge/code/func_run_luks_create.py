from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def run_luks_create(self, device, keyfile, passphrase, keyslot, keysize, cipher, hash_, sector_size, pbkdf):
    luks_type = self._module.params['type']
    label = self._module.params['label']
    options = []
    if keysize is not None:
        options.append('--key-size=' + str(keysize))
    if label is not None:
        options.extend(['--label', label])
        luks_type = 'luks2'
    if luks_type is not None:
        options.extend(['--type', luks_type])
    if cipher is not None:
        options.extend(['--cipher', cipher])
    if hash_ is not None:
        options.extend(['--hash', hash_])
    if pbkdf is not None:
        self._add_pbkdf_options(options, pbkdf)
    if sector_size is not None:
        options.extend(['--sector-size', str(sector_size)])
    if keyslot is not None:
        options.extend(['--key-slot', str(keyslot)])
    args = [self._cryptsetup_bin, 'luksFormat']
    args.extend(options)
    args.extend(['-q', device])
    if keyfile:
        args.append(keyfile)
    result = self._run_command(args, data=passphrase)
    if result[RETURN_CODE] != 0:
        raise ValueError('Error while creating LUKS on %s: %s' % (device, result[STDERR]))