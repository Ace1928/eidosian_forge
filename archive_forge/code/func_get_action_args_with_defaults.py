from __future__ import (absolute_import, division, print_function)
import ast
import base64
import datetime
import json
import os
import shlex
import time
import zipfile
import re
import pkgutil
from ast import AST, Import, ImportFrom
from io import BytesIO
from ansible.release import __version__, __author__
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.executor.interpreter_discovery import InterpreterDiscoveryRequiredError
from ansible.executor.powershell import module_manifest as ps_manifest
from ansible.module_utils.common.json import AnsibleJSONEncoder
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.plugins.loader import module_utils_loader
from ansible.utils.collection_loader._collection_finder import _get_collection_metadata, _nested_dict_get
from ansible.executor import action_write_locks
from ansible.utils.display import Display
from collections import namedtuple
import importlib.util
import importlib.machinery
import sys
import {1} as mod
def get_action_args_with_defaults(action, args, defaults, templar, action_groups=None):
    if action_groups is None:
        msg = 'Finding module_defaults for action %s. The caller has not passed the action_groups, so any that may include this action will be ignored.'
        display.warning(msg=msg)
        group_names = []
    else:
        group_names = action_groups.get(action, [])
    tmp_args = {}
    module_defaults = {}
    if isinstance(defaults, list):
        for default in defaults:
            module_defaults.update(default)
    module_defaults = templar.template(module_defaults)
    for default in module_defaults:
        if default.startswith('group/'):
            group_name = default.split('group/')[-1]
            if group_name in group_names:
                tmp_args.update((module_defaults.get('group/%s' % group_name) or {}).copy())
    tmp_args.update(module_defaults.get(action, {}).copy())
    tmp_args.update(args)
    return tmp_args