from __future__ import annotations
import runpy
import inspect
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from ansible.executor.powershell.module_manifest import PSModuleDepFinder
from ansible.module_utils.basic import FILE_COMMON_ARGUMENTS, AnsibleModule
from ansible.module_utils.six import reraise
from ansible.module_utils.common.text.converters import to_bytes, to_text
from .utils import CaptureStd, find_executable, get_module_name_from_filename
def get_ps_argument_spec(filename, collection):
    fqc_name = get_module_name_from_filename(filename, collection)
    pwsh = find_executable('pwsh')
    if not pwsh:
        raise FileNotFoundError('Required program for PowerShell arg spec inspection "pwsh" not found.')
    module_path = os.path.join(os.getcwd(), filename)
    b_module_path = to_bytes(module_path, errors='surrogate_or_strict')
    with open(b_module_path, mode='rb') as module_fd:
        b_module_data = module_fd.read()
    ps_dep_finder = PSModuleDepFinder()
    ps_dep_finder.scan_module(b_module_data, fqn=fqc_name)
    ps_dep_finder._add_module(name=b'Ansible.ModuleUtils.AddType', ext='.psm1', fqn=None, optional=False, wrapper=False)
    util_manifest = json.dumps({'module_path': to_text(module_path, errors='surrogate_or_strict'), 'ansible_basic': ps_dep_finder.cs_utils_module['Ansible.Basic']['path'], 'ps_utils': {name: info['path'] for name, info in ps_dep_finder.ps_modules.items()}})
    script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ps_argspec.ps1')
    proc = subprocess.run(['pwsh', script_path, util_manifest], stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AnsibleModuleImportError('STDOUT:\n%s\nSTDERR:\n%s' % (proc.stdout, proc.stderr))
    kwargs = json.loads(proc.stdout)
    kwargs['argument_spec'] = kwargs.pop('options', {})
    return (kwargs['argument_spec'], kwargs)