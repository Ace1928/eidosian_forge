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
def get_py_argument_spec(filename, collection):
    name = get_module_name_from_filename(filename, collection)
    with setup_env(filename) as fake:
        try:
            with CaptureStd():
                runpy.run_module(name, run_name='__main__', alter_sys=True)
        except AnsibleModuleCallError:
            pass
        except BaseException as e:
            reraise(AnsibleModuleImportError, AnsibleModuleImportError('%s' % e), sys.exc_info()[2])
        if not fake.called:
            raise AnsibleModuleNotInitialized()
    try:
        for arg, arg_name in zip(fake.args, ANSIBLE_MODULE_CONSTRUCTOR_ARGS):
            fake.kwargs[arg_name] = arg
        argument_spec = fake.kwargs.get('argument_spec') or {}
        if fake.kwargs.get('add_file_common_args'):
            for k, v in FILE_COMMON_ARGUMENTS.items():
                if k not in argument_spec:
                    argument_spec[k] = v
        return (argument_spec, fake.kwargs)
    except (TypeError, IndexError):
        return ({}, {})