from __future__ import (absolute_import, division, print_function)
import locale
import os
import sys
from importlib.metadata import version
from ansible.module_utils.compat.version import LooseVersion
import errno
import getpass
import subprocess
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError
from ansible.inventory.manager import InventoryManager
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.file import is_executable
from ansible.parsing.dataloader import DataLoader
from ansible.parsing.vault import PromptVaultSecret, get_file_vault_secret
from ansible.plugins.loader import add_all_plugin_dirs, init_plugin_loader
from ansible.release import __version__
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path
from ansible.utils.path import unfrackpath
from ansible.utils.unsafe_proxy import to_unsafe_text
from ansible.vars.manager import VariableManager
@staticmethod
def get_password_from_file(pwd_file):
    b_pwd_file = to_bytes(pwd_file)
    secret = None
    if b_pwd_file == b'-':
        secret = sys.stdin.buffer.read()
    elif not os.path.exists(b_pwd_file):
        raise AnsibleError('The password file %s was not found' % pwd_file)
    elif is_executable(b_pwd_file):
        display.vvvv(u'The password file %s is a script.' % to_text(pwd_file))
        cmd = [b_pwd_file]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            raise AnsibleError('Problem occured when trying to run the password script %s (%s). If this is not a script, remove the executable bit from the file.' % (pwd_file, e))
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            raise AnsibleError('The password script %s returned an error (rc=%s): %s' % (pwd_file, p.returncode, stderr))
        secret = stdout
    else:
        try:
            f = open(b_pwd_file, 'rb')
            secret = f.read().strip()
            f.close()
        except (OSError, IOError) as e:
            raise AnsibleError('Could not read password file %s: %s' % (pwd_file, e))
    secret = secret.strip(b'\r\n')
    if not secret:
        raise AnsibleError('Empty password was provided from file (%s)' % pwd_file)
    return to_unsafe_text(secret)