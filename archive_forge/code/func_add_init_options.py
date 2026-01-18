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
def add_init_options(self, parser, parents=None):
    galaxy_type = 'collection' if parser.metavar == 'COLLECTION_ACTION' else 'role'
    init_parser = parser.add_parser('init', parents=parents, help='Initialize new {0} with the base structure of a {0}.'.format(galaxy_type))
    init_parser.set_defaults(func=self.execute_init)
    init_parser.add_argument('--init-path', dest='init_path', default='./', help='The path in which the skeleton {0} will be created. The default is the current working directory.'.format(galaxy_type))
    init_parser.add_argument('--{0}-skeleton'.format(galaxy_type), dest='{0}_skeleton'.format(galaxy_type), default=C.GALAXY_COLLECTION_SKELETON if galaxy_type == 'collection' else C.GALAXY_ROLE_SKELETON, help='The path to a {0} skeleton that the new {0} should be based upon.'.format(galaxy_type))
    obj_name_kwargs = {}
    if galaxy_type == 'collection':
        obj_name_kwargs['type'] = validate_collection_name
    init_parser.add_argument('{0}_name'.format(galaxy_type), help='{0} name'.format(galaxy_type.capitalize()), **obj_name_kwargs)
    if galaxy_type == 'role':
        init_parser.add_argument('--type', dest='role_type', action='store', default='default', help="Initialize using an alternate role type. Valid types include: 'container', 'apb' and 'network'.")