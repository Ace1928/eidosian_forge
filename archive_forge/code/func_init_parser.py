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
@abstractmethod
def init_parser(self, usage='', desc=None, epilog=None):
    """
        Create an options parser for most ansible scripts

        Subclasses need to implement this method.  They will usually call the base class's
        init_parser to create a basic version and then add their own options on top of that.

        An implementation will look something like this::

            def init_parser(self):
                super(MyCLI, self).init_parser(usage="My Ansible CLI", inventory_opts=True)
                ansible.arguments.option_helpers.add_runas_options(self.parser)
                self.parser.add_option('--my-option', dest='my_option', action='store')
        """
    self.parser = opt_help.create_base_parser(self.name, usage=usage, desc=desc, epilog=epilog)