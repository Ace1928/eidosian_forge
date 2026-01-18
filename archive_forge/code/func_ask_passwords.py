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
def ask_passwords():
    """ prompt for connection and become passwords if needed """
    op = context.CLIARGS
    sshpass = None
    becomepass = None
    become_prompt = ''
    become_prompt_method = 'BECOME' if C.AGNOSTIC_BECOME_PROMPT else op['become_method'].upper()
    try:
        become_prompt = '%s password: ' % become_prompt_method
        if op['ask_pass']:
            sshpass = CLI._get_secret('SSH password: ')
            become_prompt = '%s password[defaults to SSH password]: ' % become_prompt_method
        elif op['connection_password_file']:
            sshpass = CLI.get_password_from_file(op['connection_password_file'])
        if op['become_ask_pass']:
            becomepass = CLI._get_secret(become_prompt)
            if op['ask_pass'] and becomepass == '':
                becomepass = sshpass
        elif op['become_password_file']:
            becomepass = CLI.get_password_from_file(op['become_password_file'])
    except EOFError:
        pass
    return (sshpass, becomepass)