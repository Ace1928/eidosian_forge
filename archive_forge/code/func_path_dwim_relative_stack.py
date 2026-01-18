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
def path_dwim_relative_stack(self, paths: list[str], dirname: str, source: str, is_role: bool=False) -> str:
    """
        find one file in first path in stack taking roles into account and adding play basedir as fallback

        :arg paths: A list of text strings which are the paths to look for the filename in.
        :arg dirname: A text string representing a directory.  The directory
            is prepended to the source to form the path to search for.
        :arg source: A text string which is the filename to search for
        :rtype: A text string
        :returns: An absolute path to the filename ``source`` if found
        :raises: An AnsibleFileNotFound Exception if the file is found to exist in the search paths
        """
    b_dirname = to_bytes(dirname, errors='surrogate_or_strict')
    b_source = to_bytes(source, errors='surrogate_or_strict')
    result = None
    search = []
    if source is None:
        display.warning('Invalid request to find a file that matches a "null" value')
    elif source and (source.startswith('~') or source.startswith(os.path.sep)):
        test_path = unfrackpath(b_source, follow=False)
        if os.path.exists(to_bytes(test_path, errors='surrogate_or_strict')):
            result = test_path
    else:
        display.debug(u'evaluation_path:\n\t%s' % '\n\t'.join(paths))
        for path in paths:
            upath = unfrackpath(path, follow=False)
            b_upath = to_bytes(upath, errors='surrogate_or_strict')
            b_pb_base_dir = os.path.dirname(b_upath)
            if (is_role or self._is_role(path)) and b_pb_base_dir.endswith(b'/tasks'):
                search.append(os.path.join(os.path.dirname(b_pb_base_dir), b_dirname, b_source))
                search.append(os.path.join(b_pb_base_dir, b_source))
            else:
                if b_source.split(b'/')[0] != dirname:
                    search.append(os.path.join(b_upath, b_dirname, b_source))
                search.append(os.path.join(b_upath, b_source))
        if b_source.split(b'/')[0] != dirname:
            search.append(os.path.join(to_bytes(self.get_basedir(), errors='surrogate_or_strict'), b_dirname, b_source))
        search.append(os.path.join(to_bytes(self.get_basedir(), errors='surrogate_or_strict'), b_source))
        display.debug(u'search_path:\n\t%s' % to_text(b'\n\t'.join(search)))
        for b_candidate in search:
            display.vvvvv(u'looking for "%s" at "%s"' % (source, to_text(b_candidate)))
            if os.path.exists(b_candidate):
                result = to_text(b_candidate)
                break
    if result is None:
        raise AnsibleFileNotFound(file_name=source, paths=[to_native(p) for p in search])
    return result