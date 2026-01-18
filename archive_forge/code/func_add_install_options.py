from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import argparse
import functools
import json
import os.path
import pathlib
import re
import shutil
import sys
import textwrap
import time
import typing as t
from dataclasses import dataclass
from yaml.error import YAMLError
import ansible.constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.galaxy import Galaxy, get_collections_galaxy_meta_info
from ansible.galaxy.api import GalaxyAPI, GalaxyError
from ansible.galaxy.collection import (
from ansible.galaxy.collection.concrete_artifact_manager import (
from ansible.galaxy.collection.gpg import GPG_ERROR_MAP
from ansible.galaxy.dependency_resolution.dataclasses import Requirement
from ansible.galaxy.role import GalaxyRole
from ansible.galaxy.token import BasicAuthToken, GalaxyToken, KeycloakToken, NoTokenSentinel
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.yaml import yaml_dump, yaml_load
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils import six
from ansible.parsing.dataloader import DataLoader
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.playbook.role.requirement import RoleRequirement
from ansible.template import Templar
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.display import Display
from ansible.utils.plugin_docs import get_versioned_doclink
def add_install_options(self, parser, parents=None):
    galaxy_type = 'collection' if parser.metavar == 'COLLECTION_ACTION' else 'role'
    args_kwargs = {}
    if galaxy_type == 'collection':
        args_kwargs['help'] = 'The collection(s) name or path/url to a tar.gz collection artifact. This is mutually exclusive with --requirements-file.'
        ignore_errors_help = 'Ignore errors during installation and continue with the next specified collection. This will not ignore dependency conflict errors.'
    else:
        args_kwargs['help'] = 'Role name, URL or tar file'
        ignore_errors_help = 'Ignore errors and continue with the next specified role.'
    install_parser = parser.add_parser('install', parents=parents, help='Install {0}(s) from file(s), URL(s) or Ansible Galaxy'.format(galaxy_type))
    install_parser.set_defaults(func=self.execute_install)
    install_parser.add_argument('args', metavar='{0}_name'.format(galaxy_type), nargs='*', **args_kwargs)
    install_parser.add_argument('-i', '--ignore-errors', dest='ignore_errors', action='store_true', default=False, help=ignore_errors_help)
    install_exclusive = install_parser.add_mutually_exclusive_group()
    install_exclusive.add_argument('-n', '--no-deps', dest='no_deps', action='store_true', default=False, help="Don't download {0}s listed as dependencies.".format(galaxy_type))
    install_exclusive.add_argument('--force-with-deps', dest='force_with_deps', action='store_true', default=False, help='Force overwriting an existing {0} and its dependencies.'.format(galaxy_type))
    valid_signature_count_help = 'The number of signatures that must successfully verify the collection. This should be a positive integer or -1 to signify that all signatures must be used to verify the collection. Prepend the value with + to fail if no valid signatures are found for the collection (e.g. +all).'
    ignore_gpg_status_help = 'A space separated list of status codes to ignore during signature verification (for example, NO_PUBKEY FAILURE). Descriptions for the choices can be seen at L(https://github.com/gpg/gnupg/blob/master/doc/DETAILS#general-status-codes).Note: specify these after positional arguments or use -- to separate them.'
    if galaxy_type == 'collection':
        install_parser.add_argument('-p', '--collections-path', dest='collections_path', default=self._get_default_collection_path(), help='The path to the directory containing your collections.')
        install_parser.add_argument('-r', '--requirements-file', dest='requirements', help='A file containing a list of collections to be installed.')
        install_parser.add_argument('--pre', dest='allow_pre_release', action='store_true', help='Include pre-release versions. Semantic versioning pre-releases are ignored by default')
        install_parser.add_argument('-U', '--upgrade', dest='upgrade', action='store_true', default=False, help='Upgrade installed collection artifacts. This will also update dependencies unless --no-deps is provided')
        install_parser.add_argument('--keyring', dest='keyring', default=C.GALAXY_GPG_KEYRING, help='The keyring used during signature verification')
        install_parser.add_argument('--disable-gpg-verify', dest='disable_gpg_verify', action='store_true', default=C.GALAXY_DISABLE_GPG_VERIFY, help='Disable GPG signature verification when installing collections from a Galaxy server')
        install_parser.add_argument('--signature', dest='signatures', action='append', help='An additional signature source to verify the authenticity of the MANIFEST.json before installing the collection from a Galaxy server. Use in conjunction with a positional collection name (mutually exclusive with --requirements-file).')
        install_parser.add_argument('--required-valid-signature-count', dest='required_valid_signature_count', type=validate_signature_count, help=valid_signature_count_help, default=C.GALAXY_REQUIRED_VALID_SIGNATURE_COUNT)
        install_parser.add_argument('--ignore-signature-status-code', dest='ignore_gpg_errors', type=str, action='append', help=opt_help.argparse.SUPPRESS, default=C.GALAXY_IGNORE_INVALID_SIGNATURE_STATUS_CODES, choices=list(GPG_ERROR_MAP.keys()))
        install_parser.add_argument('--ignore-signature-status-codes', dest='ignore_gpg_errors', type=str, action='extend', nargs='+', help=ignore_gpg_status_help, default=C.GALAXY_IGNORE_INVALID_SIGNATURE_STATUS_CODES, choices=list(GPG_ERROR_MAP.keys()))
        install_parser.add_argument('--offline', dest='offline', action='store_true', default=False, help='Install collection artifacts (tarballs) without contacting any distribution servers. This does not apply to collections in remote Git repositories or URLs to remote tarballs.')
    else:
        install_parser.add_argument('-r', '--role-file', dest='requirements', help='A file containing a list of roles to be installed.')
        r_re = re.compile('^(?<!-)-[a-zA-Z]*r[a-zA-Z]*')
        contains_r = bool([a for a in self._raw_args if r_re.match(a)])
        role_file_re = re.compile('--role-file($|=)')
        contains_role_file = bool([a for a in self._raw_args if role_file_re.match(a)])
        if self._implicit_role and (contains_r or contains_role_file):
            install_parser.add_argument('--keyring', dest='keyring', default=C.GALAXY_GPG_KEYRING, help='The keyring used during collection signature verification')
            install_parser.add_argument('--disable-gpg-verify', dest='disable_gpg_verify', action='store_true', default=C.GALAXY_DISABLE_GPG_VERIFY, help='Disable GPG signature verification when installing collections from a Galaxy server')
            install_parser.add_argument('--required-valid-signature-count', dest='required_valid_signature_count', type=validate_signature_count, help=valid_signature_count_help, default=C.GALAXY_REQUIRED_VALID_SIGNATURE_COUNT)
            install_parser.add_argument('--ignore-signature-status-code', dest='ignore_gpg_errors', type=str, action='append', help=opt_help.argparse.SUPPRESS, default=C.GALAXY_IGNORE_INVALID_SIGNATURE_STATUS_CODES, choices=list(GPG_ERROR_MAP.keys()))
            install_parser.add_argument('--ignore-signature-status-codes', dest='ignore_gpg_errors', type=str, action='extend', nargs='+', help=ignore_gpg_status_help, default=C.GALAXY_IGNORE_INVALID_SIGNATURE_STATUS_CODES, choices=list(GPG_ERROR_MAP.keys()))
        install_parser.add_argument('-g', '--keep-scm-meta', dest='keep_scm_meta', action='store_true', default=False, help='Use tar instead of the scm archive option when packaging the role.')