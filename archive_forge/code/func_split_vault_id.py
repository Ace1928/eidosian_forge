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
def split_vault_id(vault_id):
    if '@' not in vault_id:
        return (None, vault_id)
    parts = vault_id.split('@', 1)
    ret = tuple(parts)
    return ret