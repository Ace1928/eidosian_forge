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
def execute_verify(self, artifacts_manager=None):
    """Compare checksums with the collection(s) found on the server and the installed copy. This does not verify dependencies."""
    collections = context.CLIARGS['args']
    search_paths = AnsibleCollectionConfig.collection_paths
    ignore_errors = context.CLIARGS['ignore_errors']
    local_verify_only = context.CLIARGS['offline']
    requirements_file = context.CLIARGS['requirements']
    signatures = context.CLIARGS['signatures']
    if signatures is not None:
        signatures = list(signatures)
    requirements = self._require_one_of_collections_requirements(collections, requirements_file, signatures=signatures, artifacts_manager=artifacts_manager)['collections']
    resolved_paths = [validate_collection_path(GalaxyCLI._resolve_path(path)) for path in search_paths]
    results = verify_collections(requirements, resolved_paths, self.api_servers, ignore_errors, local_verify_only=local_verify_only, artifacts_manager=artifacts_manager)
    if any((result for result in results if not result.success)):
        return 1
    return 0