from __future__ import (absolute_import, division, print_function)
import errno
import fcntl
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import warnings
from binascii import hexlify
from binascii import unhexlify
from binascii import Error as BinasciiError
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible import constants as C
from ansible.module_utils.six import binary_type
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.utils.display import Display
from ansible.utils.path import makedirs_safe, unfrackpath
def decrypt_and_get_vault_id(self, vaulttext, filename=None, obj=None):
    """Decrypt a piece of vault encrypted data.

        :arg vaulttext: a string to decrypt.  Since vault encrypted data is an
            ascii text format this can be either a byte str or unicode string.
        :kwarg filename: a filename that the data came from.  This is only
            used to make better error messages in case the data cannot be
            decrypted.
        :returns: a byte string containing the decrypted data and the vault-id vault-secret that was used

        """
    b_vaulttext = to_bytes(vaulttext, errors='strict', encoding='utf-8')
    if self.secrets is None:
        msg = 'A vault password must be specified to decrypt data'
        if filename:
            msg += ' in file %s' % to_native(filename)
        raise AnsibleVaultError(msg)
    if not is_encrypted(b_vaulttext):
        msg = 'input is not vault encrypted data. '
        if filename:
            msg += '%s is not a vault encrypted file' % to_native(filename)
        raise AnsibleError(msg)
    b_vaulttext, dummy, cipher_name, vault_id = parse_vaulttext_envelope(b_vaulttext, filename=filename)
    if cipher_name in CIPHER_WHITELIST:
        this_cipher = CIPHER_MAPPING[cipher_name]()
    else:
        raise AnsibleError('{0} cipher could not be found'.format(cipher_name))
    b_plaintext = None
    if not self.secrets:
        raise AnsibleVaultError('Attempting to decrypt but no vault secrets found')
    vault_id_matchers = []
    vault_id_used = None
    vault_secret_used = None
    if vault_id:
        display.vvvvv(u'Found a vault_id (%s) in the vaulttext' % to_text(vault_id))
        vault_id_matchers.append(vault_id)
        _matches = match_secrets(self.secrets, vault_id_matchers)
        if _matches:
            display.vvvvv(u'We have a secret associated with vault id (%s), will try to use to decrypt %s' % (to_text(vault_id), to_text(filename)))
        else:
            display.vvvvv(u'Found a vault_id (%s) in the vault text, but we do not have a associated secret (--vault-id)' % to_text(vault_id))
    if not C.DEFAULT_VAULT_ID_MATCH:
        vault_id_matchers.extend([_vault_id for _vault_id, _dummy in self.secrets if _vault_id != vault_id])
    matched_secrets = match_secrets(self.secrets, vault_id_matchers)
    for vault_secret_id, vault_secret in matched_secrets:
        display.vvvvv(u'Trying to use vault secret=(%s) id=%s to decrypt %s' % (to_text(vault_secret), to_text(vault_secret_id), to_text(filename)))
        try:
            display.vvvv(u'Trying secret %s for vault_id=%s' % (to_text(vault_secret), to_text(vault_secret_id)))
            b_plaintext = this_cipher.decrypt(b_vaulttext, vault_secret)
            if b_plaintext is not None:
                vault_id_used = vault_secret_id
                vault_secret_used = vault_secret
                file_slug = ''
                if filename:
                    file_slug = ' of "%s"' % filename
                display.vvvvv(u'Decrypt%s successful with secret=%s and vault_id=%s' % (to_text(file_slug), to_text(vault_secret), to_text(vault_secret_id)))
                break
        except AnsibleVaultFormatError as exc:
            exc.obj = obj
            msg = u'There was a vault format error'
            if filename:
                msg += u' in %s' % to_text(filename)
            msg += u': %s' % to_text(exc)
            display.warning(msg, formatted=True)
            raise
        except AnsibleError as e:
            display.vvvv(u'Tried to use the vault secret (%s) to decrypt (%s) but it failed. Error: %s' % (to_text(vault_secret_id), to_text(filename), e))
            continue
    else:
        msg = 'Decryption failed (no vault secrets were found that could decrypt)'
        if filename:
            msg += ' on %s' % to_native(filename)
        raise AnsibleVaultError(msg)
    if b_plaintext is None:
        msg = 'Decryption failed'
        if filename:
            msg += ' on %s' % to_native(filename)
        raise AnsibleError(msg)
    return (b_plaintext, vault_id_used, vault_secret_used)