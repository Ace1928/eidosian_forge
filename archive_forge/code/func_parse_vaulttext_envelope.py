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
def parse_vaulttext_envelope(b_vaulttext_envelope, default_vault_id=None, filename=None):
    """Parse the vaulttext envelope

    When data is saved, it has a header prepended and is formatted into 80
    character lines.  This method extracts the information from the header
    and then removes the header and the inserted newlines.  The string returned
    is suitable for processing by the Cipher classes.

    :arg b_vaulttext: byte str containing the data from a save file
    :kwarg default_vault_id: The vault_id name to use if the vaulttext does not provide one.
    :kwarg filename: The filename that the data came from.  This is only
        used to make better error messages in case the data cannot be
        decrypted. This is optional.
    :returns: A tuple of byte str of the vaulttext suitable to pass to parse_vaultext,
        a byte str of the vault format version,
        the name of the cipher used, and the vault_id.
    :raises: AnsibleVaultFormatError: if the vaulttext_envelope format is invalid
    """
    default_vault_id = default_vault_id or C.DEFAULT_VAULT_IDENTITY
    try:
        return _parse_vaulttext_envelope(b_vaulttext_envelope, default_vault_id)
    except Exception as exc:
        msg = 'Vault envelope format error'
        if filename:
            msg += ' in %s' % filename
        msg += ': %s' % exc
        raise AnsibleVaultFormatError(msg)