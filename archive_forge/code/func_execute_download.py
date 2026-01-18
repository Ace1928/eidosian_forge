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
def execute_download(self, artifacts_manager=None):
    """Download collections and their dependencies as a tarball for an offline install."""
    collections = context.CLIARGS['args']
    no_deps = context.CLIARGS['no_deps']
    download_path = context.CLIARGS['download_path']
    requirements_file = context.CLIARGS['requirements']
    if requirements_file:
        requirements_file = GalaxyCLI._resolve_path(requirements_file)
    requirements = self._require_one_of_collections_requirements(collections, requirements_file, artifacts_manager=artifacts_manager)['collections']
    download_path = GalaxyCLI._resolve_path(download_path)
    b_download_path = to_bytes(download_path, errors='surrogate_or_strict')
    if not os.path.exists(b_download_path):
        os.makedirs(b_download_path)
    download_collections(requirements, download_path, self.api_servers, no_deps, context.CLIARGS['allow_pre_release'], artifacts_manager=artifacts_manager)
    return 0