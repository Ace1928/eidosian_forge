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
def recursive_finder(name, module_fqn, module_data, zf, date_time=None):
    """
    Using ModuleDepFinder, make sure we have all of the module_utils files that
    the module and its module_utils files needs. (no longer actually recursive)
    :arg name: Name of the python module we're examining
    :arg module_fqn: Fully qualified name of the python module we're scanning
    :arg module_data: string Python code of the module we're scanning
    :arg zf: An open :python:class:`zipfile.ZipFile` object that holds the Ansible module payload
        which we're assembling
    """
    if date_time is None:
        date_time = time.gmtime()[:6]
    py_module_cache = {('ansible',): (b'from pkgutil import extend_path\n__path__=extend_path(__path__,__name__)\n__version__="' + to_bytes(__version__) + b'"\n__author__="' + to_bytes(__author__) + b'"\n', 'ansible/__init__.py'), ('ansible', 'module_utils'): (b'from pkgutil import extend_path\n__path__=extend_path(__path__,__name__)\n', 'ansible/module_utils/__init__.py')}
    module_utils_paths = [p for p in module_utils_loader._get_paths(subdirs=False) if os.path.isdir(p)]
    module_utils_paths.append(_MODULE_UTILS_PATH)
    try:
        tree = compile(module_data, '<unknown>', 'exec', ast.PyCF_ONLY_AST)
    except (SyntaxError, IndentationError) as e:
        raise AnsibleError('Unable to import %s due to %s' % (name, e.msg))
    finder = ModuleDepFinder(module_fqn, tree)
    modules_to_process = [ModuleUtilsProcessEntry(m, True, False, is_optional=m in finder.optional_imports) for m in finder.submodules]
    modules_to_process.append(ModuleUtilsProcessEntry(('ansible', 'module_utils', 'basic'), False, False, is_optional=False))
    while modules_to_process:
        modules_to_process.sort()
        py_module_name, is_ambiguous, child_is_redirected, is_optional = modules_to_process.pop(0)
        if py_module_name in py_module_cache:
            continue
        if py_module_name[0:2] == ('ansible', 'module_utils'):
            module_info = LegacyModuleUtilLocator(py_module_name, is_ambiguous=is_ambiguous, mu_paths=module_utils_paths, child_is_redirected=child_is_redirected)
        elif py_module_name[0] == 'ansible_collections':
            module_info = CollectionModuleUtilLocator(py_module_name, is_ambiguous=is_ambiguous, child_is_redirected=child_is_redirected, is_optional=is_optional)
        else:
            display.warning('ModuleDepFinder improperly found a non-module_utils import %s' % [py_module_name])
            continue
        if not module_info.found:
            if is_optional:
                continue
            msg = 'Could not find imported module support code for {0}.  Looked for ({1})'.format(module_fqn, module_info.candidate_names_joined)
            raise AnsibleError(msg)
        if module_info.fq_name_parts in py_module_cache:
            continue
        try:
            tree = compile(module_info.source_code, '<unknown>', 'exec', ast.PyCF_ONLY_AST)
        except (SyntaxError, IndentationError) as e:
            raise AnsibleError('Unable to import %s due to %s' % (module_info.fq_name_parts, e.msg))
        finder = ModuleDepFinder('.'.join(module_info.fq_name_parts), tree, module_info.is_package)
        modules_to_process.extend((ModuleUtilsProcessEntry(m, True, False, is_optional=m in finder.optional_imports) for m in finder.submodules if m not in py_module_cache))
        py_module_cache[module_info.fq_name_parts] = (module_info.source_code, module_info.output_path)
        accumulated_pkg_name = []
        for pkg in module_info.fq_name_parts[:-1]:
            accumulated_pkg_name.append(pkg)
            normalized_name = tuple(accumulated_pkg_name)
            if normalized_name not in py_module_cache:
                modules_to_process.append(ModuleUtilsProcessEntry(normalized_name, False, module_info.redirected, is_optional=is_optional))
    for py_module_name in py_module_cache:
        py_module_file_name = py_module_cache[py_module_name][1]
        zf.writestr(_make_zinfo(py_module_file_name, date_time, zf=zf), py_module_cache[py_module_name][0])
        mu_file = to_text(py_module_file_name, errors='surrogate_or_strict')
        display.vvvvv('Including module_utils file %s' % mu_file)