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
def execute_encrypt(self):
    """ encrypt the supplied file using the provided vault secret """
    if not context.CLIARGS['args'] and sys.stdin.isatty():
        display.display('Reading plaintext input from stdin', stderr=True)
    for f in context.CLIARGS['args'] or ['-']:
        self.editor.encrypt_file(f, self.encrypt_secret, vault_id=self.encrypt_vault_id, output_file=context.CLIARGS['output_file'])
    if sys.stdout.isatty():
        display.display('Encryption successful', stderr=True)