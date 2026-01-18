from __future__ import (absolute_import, division, print_function)
import copy
import os
import os.path
import re
import tempfile
import typing as t
from ansible import constants as C
from ansible.errors import AnsibleFileNotFound, AnsibleParserError
from ansible.module_utils.basic import is_executable
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.parsing.quoting import unquote
from ansible.parsing.utils.yaml import from_yaml
from ansible.parsing.vault import VaultLib, b_HEADER, is_encrypted, is_encrypted_file, parse_vaulttext_envelope, PromptVaultSecret
from ansible.utils.path import unfrackpath
from ansible.utils.display import Display
def get_real_file(self, file_path: str, decrypt: bool=True) -> str:
    """
        If the file is vault encrypted return a path to a temporary decrypted file
        If the file is not encrypted then the path is returned
        Temporary files are cleanup in the destructor
        """
    if not file_path or not isinstance(file_path, (binary_type, text_type)):
        raise AnsibleParserError("Invalid filename: '%s'" % to_native(file_path))
    b_file_path = to_bytes(file_path, errors='surrogate_or_strict')
    if not self.path_exists(b_file_path) or not self.is_file(b_file_path):
        raise AnsibleFileNotFound(file_name=file_path)
    real_path = self.path_dwim(file_path)
    try:
        if decrypt:
            with open(to_bytes(real_path), 'rb') as f:
                if is_encrypted_file(f, count=len(b_HEADER)):
                    data = f.read()
                    if not self._vault.secrets:
                        raise AnsibleParserError('A vault password or secret must be specified to decrypt %s' % to_native(file_path))
                    data = self._vault.decrypt(data, filename=real_path)
                    real_path = self._create_content_tempfile(data)
                    self._tempfiles.add(real_path)
        return real_path
    except (IOError, OSError) as e:
        raise AnsibleParserError("an error occurred while trying to read the file '%s': %s" % (to_native(real_path), to_native(e)), orig_exc=e)