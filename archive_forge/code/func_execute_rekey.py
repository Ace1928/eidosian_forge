from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import os
import sys
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_text, to_bytes
from ansible.parsing.dataloader import DataLoader
from ansible.parsing.vault import VaultEditor, VaultLib, match_encrypt_secret
from ansible.utils.display import Display
def execute_rekey(self):
    """ re-encrypt a vaulted file with a new secret, the previous secret is required """
    for f in context.CLIARGS['args']:
        self.editor.rekey_file(f, self.new_encrypt_secret, self.new_encrypt_vault_id)
    display.display('Rekey successful', stderr=True)