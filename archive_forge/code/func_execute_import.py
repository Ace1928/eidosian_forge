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
def execute_import(self):
    """ used to import a role into Ansible Galaxy """
    colors = {'INFO': 'normal', 'WARNING': C.COLOR_WARN, 'ERROR': C.COLOR_ERROR, 'SUCCESS': C.COLOR_OK, 'FAILED': C.COLOR_ERROR}
    github_user = to_text(context.CLIARGS['github_user'], errors='surrogate_or_strict')
    github_repo = to_text(context.CLIARGS['github_repo'], errors='surrogate_or_strict')
    rc = 0
    if context.CLIARGS['check_status']:
        task = self.api.get_import_task(github_user=github_user, github_repo=github_repo)
    else:
        task = self.api.create_import_task(github_user, github_repo, reference=context.CLIARGS['reference'], role_name=context.CLIARGS['role_name'])
        if len(task) > 1:
            display.display('WARNING: More than one Galaxy role associated with Github repo %s/%s.' % (github_user, github_repo), color='yellow')
            display.display('The following Galaxy roles are being updated:' + u'\n', color=C.COLOR_CHANGED)
            for t in task:
                display.display('%s.%s' % (t['summary_fields']['role']['namespace'], t['summary_fields']['role']['name']), color=C.COLOR_CHANGED)
            display.display(u'\nTo properly namespace this role, remove each of the above and re-import %s/%s from scratch' % (github_user, github_repo), color=C.COLOR_CHANGED)
            return rc
        display.display('Successfully submitted import request %d' % task[0]['id'])
        if not context.CLIARGS['wait']:
            display.display('Role name: %s' % task[0]['summary_fields']['role']['name'])
            display.display('Repo: %s/%s' % (task[0]['github_user'], task[0]['github_repo']))
    if context.CLIARGS['check_status'] or context.CLIARGS['wait']:
        msg_list = []
        finished = False
        while not finished:
            task = self.api.get_import_task(task_id=task[0]['id'])
            for msg in task[0]['summary_fields']['task_messages']:
                if msg['id'] not in msg_list:
                    display.display(msg['message_text'], color=colors[msg['message_type']])
                    msg_list.append(msg['id'])
            if (state := task[0]['state']) in ['SUCCESS', 'FAILED']:
                rc = ['SUCCESS', 'FAILED'].index(state)
                finished = True
            else:
                time.sleep(10)
    return rc