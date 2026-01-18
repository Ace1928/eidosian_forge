from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_custom_target_shell_build_phases(self, objects_dict: PbxDict) -> None:
    for tname, t in self.custom_targets.items():
        if not isinstance(t, build.CustomTarget):
            continue
        srcs, ofilenames, cmd = self.eval_custom_target_command(t, absolute_outputs=True)
        fixed_cmd, _ = self.as_meson_exe_cmdline(cmd[0], cmd[1:], capture=ofilenames[0] if t.capture else None, feed=srcs[0] if t.feed else None, env=t.env)
        custom_dict = PbxDict()
        objects_dict.add_item(self.shell_targets[tname], custom_dict, f'/* Custom target {tname} */')
        custom_dict.add_item('isa', 'PBXShellScriptBuildPhase')
        custom_dict.add_item('buildActionMask', 2147483647)
        custom_dict.add_item('files', PbxArray())
        custom_dict.add_item('inputPaths', PbxArray())
        outarray = PbxArray()
        custom_dict.add_item('name', '"Generate {}."'.format(ofilenames[0]))
        custom_dict.add_item('outputPaths', outarray)
        for o in ofilenames:
            outarray.add_item(f'"{os.path.join(self.environment.get_build_dir(), o)}"')
        custom_dict.add_item('runOnlyForDeploymentPostprocessing', 0)
        custom_dict.add_item('shellPath', '/bin/sh')
        workdir = self.environment.get_build_dir()
        quoted_cmd = []
        for c in fixed_cmd:
            quoted_cmd.append(c.replace('"', chr(92) + '"'))
        cmdstr = ' '.join([f"\\'{x}\\'" for x in quoted_cmd])
        custom_dict.add_item('shellScript', f'''"cd '{workdir}'; {cmdstr}"''')
        custom_dict.add_item('showEnvVarsInLog', 0)