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
def execute_create(self):
    """ create and open a file in an editor that will be encrypted with the provided vault secret when closed"""
    if len(context.CLIARGS['args']) != 1:
        raise AnsibleOptionsError('ansible-vault create can take only one filename argument')
    if sys.stdout.isatty() or context.CLIARGS['skip_tty_check']:
        self.editor.create_file(context.CLIARGS['args'][0], self.encrypt_secret, vault_id=self.encrypt_vault_id)
    else:
        raise AnsibleOptionsError('not a tty, editor cannot be opened')