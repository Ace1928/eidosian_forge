from __future__ import (absolute_import, division, print_function)
import copy
import operator
import argparse
import os
import os.path
import sys
import time
from jinja2 import __version__ as j2_version
import ansible
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.yaml import HAS_LIBYAML, yaml_load
from ansible.release import __version__
from ansible.utils.path import unfrackpath
def add_vault_options(parser):
    """Add options for loading vault files"""
    parser.add_argument('--vault-id', default=[], dest='vault_ids', action='append', type=str, help='the vault identity to use')
    base_group = parser.add_mutually_exclusive_group()
    base_group.add_argument('-J', '--ask-vault-password', '--ask-vault-pass', default=C.DEFAULT_ASK_VAULT_PASS, dest='ask_vault_pass', action='store_true', help='ask for vault password')
    base_group.add_argument('--vault-password-file', '--vault-pass-file', default=[], dest='vault_password_files', help='vault password file', type=unfrack_path(follow=False), action='append')