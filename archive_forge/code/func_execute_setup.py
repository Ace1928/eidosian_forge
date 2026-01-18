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
def execute_setup(self):
    """ Setup an integration from Github or Travis for Ansible Galaxy roles"""
    if context.CLIARGS['setup_list']:
        secrets = self.api.list_secrets()
        if len(secrets) == 0:
            display.display('No integrations found.')
            return 0
        display.display(u'\n' + 'ID         Source     Repo', color=C.COLOR_OK)
        display.display('---------- ---------- ----------', color=C.COLOR_OK)
        for secret in secrets:
            display.display('%-10s %-10s %s/%s' % (secret['id'], secret['source'], secret['github_user'], secret['github_repo']), color=C.COLOR_OK)
        return 0
    if context.CLIARGS['remove_id']:
        self.api.remove_secret(context.CLIARGS['remove_id'])
        display.display('Secret removed. Integrations using this secret will not longer work.', color=C.COLOR_OK)
        return 0
    source = context.CLIARGS['source']
    github_user = context.CLIARGS['github_user']
    github_repo = context.CLIARGS['github_repo']
    secret = context.CLIARGS['secret']
    resp = self.api.add_secret(source, github_user, github_repo, secret)
    display.display('Added integration for %s %s/%s' % (resp['source'], resp['github_user'], resp['github_repo']))
    return 0