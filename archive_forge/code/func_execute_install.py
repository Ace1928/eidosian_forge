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
@with_collection_artifacts_manager
def execute_install(self, artifacts_manager=None):
    """
        Install one or more roles(``ansible-galaxy role install``), or one or more collections(``ansible-galaxy collection install``).
        You can pass in a list (roles or collections) or use the file
        option listed below (these are mutually exclusive). If you pass in a list, it
        can be a name (which will be downloaded via the galaxy API and github), or it can be a local tar archive file.
        """
    install_items = context.CLIARGS['args']
    requirements_file = context.CLIARGS['requirements']
    collection_path = None
    signatures = context.CLIARGS.get('signatures')
    if signatures is not None:
        signatures = list(signatures)
    if requirements_file:
        requirements_file = GalaxyCLI._resolve_path(requirements_file)
    two_type_warning = "The requirements file '%s' contains {0}s which will be ignored. To install these {0}s run 'ansible-galaxy {0} install -r' or to install both at the same time run 'ansible-galaxy install -r' without a custom install path." % to_text(requirements_file)
    collection_requirements = []
    role_requirements = []
    if context.CLIARGS['type'] == 'collection':
        collection_path = GalaxyCLI._resolve_path(context.CLIARGS['collections_path'])
        requirements = self._require_one_of_collections_requirements(install_items, requirements_file, signatures=signatures, artifacts_manager=artifacts_manager)
        collection_requirements = requirements['collections']
        if requirements['roles']:
            display.vvv(two_type_warning.format('role'))
    else:
        if not install_items and requirements_file is None:
            raise AnsibleOptionsError('- you must specify a user/role name or a roles file')
        if requirements_file:
            if not (requirements_file.endswith('.yaml') or requirements_file.endswith('.yml')):
                raise AnsibleError('Invalid role requirements file, it must end with a .yml or .yaml extension')
            galaxy_args = self._raw_args
            will_install_collections = self._implicit_role and '-p' not in galaxy_args and ('--roles-path' not in galaxy_args)
            requirements = self._parse_requirements_file(requirements_file, artifacts_manager=artifacts_manager, validate_signature_options=will_install_collections)
            role_requirements = requirements['roles']
            if requirements['collections'] and (not self._implicit_role or '-p' in galaxy_args or '--roles-path' in galaxy_args):
                display_func = display.warning if self._implicit_role else display.vvv
                display_func(two_type_warning.format('collection'))
            else:
                collection_path = self._get_default_collection_path()
                collection_requirements = requirements['collections']
        else:
            for rname in context.CLIARGS['args']:
                role = RoleRequirement.role_yaml_parse(rname.strip())
                role_requirements.append(GalaxyRole(self.galaxy, self.lazy_role_api, **role))
    if not role_requirements and (not collection_requirements):
        display.display('Skipping install, no requirements found')
        return
    if role_requirements:
        display.display('Starting galaxy role install process')
        self._execute_install_role(role_requirements)
    if collection_requirements:
        display.display('Starting galaxy collection install process')
        self._execute_install_collection(collection_requirements, collection_path, artifacts_manager=artifacts_manager)