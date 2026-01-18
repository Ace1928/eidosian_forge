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
def parse_role_req(requirement):
    if 'include' not in requirement:
        role = RoleRequirement.role_yaml_parse(requirement)
        display.vvv('found role %s in yaml file' % to_text(role))
        if 'name' not in role and 'src' not in role:
            raise AnsibleError('Must specify name or src for role')
        return [GalaxyRole(self.galaxy, self.lazy_role_api, **role)]
    else:
        b_include_path = to_bytes(requirement['include'], errors='surrogate_or_strict')
        if not os.path.isfile(b_include_path):
            raise AnsibleError("Failed to find include requirements file '%s' in '%s'" % (to_native(b_include_path), to_native(requirements_file)))
        with open(b_include_path, 'rb') as f_include:
            try:
                return [GalaxyRole(self.galaxy, self.lazy_role_api, **r) for r in (RoleRequirement.role_yaml_parse(i) for i in yaml_load(f_include))]
            except Exception as e:
                raise AnsibleError('Unable to load data from include requirements file: %s %s' % (to_native(requirements_file), to_native(e)))