from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_test_shell_build_phase(self, objects_dict: PbxDict) -> None:
    shell_dict = PbxDict()
    objects_dict.add_item(self.test_command_id, shell_dict, 'ShellScript')
    shell_dict.add_item('isa', 'PBXShellScriptBuildPhase')
    shell_dict.add_item('buildActionMask', 2147483647)
    shell_dict.add_item('files', PbxArray())
    shell_dict.add_item('inputPaths', PbxArray())
    shell_dict.add_item('outputPaths', PbxArray())
    shell_dict.add_item('runOnlyForDeploymentPostprocessing', 0)
    shell_dict.add_item('shellPath', '/bin/sh')
    cmd = mesonlib.get_meson_command() + ['test', '--no-rebuild', '-C', self.environment.get_build_dir()]
    cmdstr = ' '.join(["'%s'" % i for i in cmd])
    shell_dict.add_item('shellScript', f'"{cmdstr}"')
    shell_dict.add_item('showEnvVarsInLog', 0)