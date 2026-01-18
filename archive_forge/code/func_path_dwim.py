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
def path_dwim(self, given: str) -> str:
    """
        make relative paths work like folks expect.
        """
    given = unquote(given)
    given = to_text(given, errors='surrogate_or_strict')
    if given.startswith(to_text(os.path.sep)) or given.startswith(u'~'):
        path = given
    else:
        basedir = to_text(self._basedir, errors='surrogate_or_strict')
        path = os.path.join(basedir, given)
    return unfrackpath(path, follow=False)