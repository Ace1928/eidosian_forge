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
def find_vars_files(self, path: str, name: str, extensions: list[str] | None=None, allow_dir: bool=True) -> list[str]:
    """
        Find vars files in a given path with specified name. This will find
        files in a dir named <name>/ or a file called <name> ending in known
        extensions.
        """
    b_path = to_bytes(os.path.join(path, name))
    found = []
    if extensions is None:
        extensions = [''] + C.YAML_FILENAME_EXTENSIONS
    for ext in extensions:
        if '.' in ext:
            full_path = b_path + to_bytes(ext)
        elif ext:
            full_path = b'.'.join([b_path, to_bytes(ext)])
        else:
            full_path = b_path
        if self.path_exists(full_path):
            if self.is_directory(full_path):
                if allow_dir:
                    found.extend(self._get_dir_vars_files(to_text(full_path), extensions))
                else:
                    continue
            else:
                found.append(to_text(full_path))
            break
    return found