from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_single_generator_phase(self, tname, t, genlist, generator_id, objects_dict) -> None:
    generator = genlist.get_generator()
    exe = generator.get_exe()
    exe_arr = self.build_target_to_cmd_array(exe)
    workdir = self.environment.get_build_dir()
    target_private_dir = self.relpath(self.get_target_private_dir(t), self.get_target_dir(t))
    gen_dict = PbxDict()
    objects_dict.add_item(self.shell_targets[tname, generator_id], gen_dict, f'"Generator {generator_id}/{tname}"')
    infilelist = genlist.get_inputs()
    outfilelist = genlist.get_outputs()
    gen_dict.add_item('isa', 'PBXShellScriptBuildPhase')
    gen_dict.add_item('buildActionMask', 2147483647)
    gen_dict.add_item('files', PbxArray())
    gen_dict.add_item('inputPaths', PbxArray())
    gen_dict.add_item('name', f'"Generator {generator_id}/{tname}"')
    commands = [['cd', workdir]]
    k = (tname, generator_id)
    ofile_abs = self.generator_outputs[k]
    outarray = PbxArray()
    gen_dict.add_item('outputPaths', outarray)
    for of in ofile_abs:
        outarray.add_item(f'"{of}"')
    for i in infilelist:
        infilename = i.rel_to_builddir(self.build_to_src, target_private_dir)
        base_args = generator.get_arglist(infilename)
        for o_base in genlist.get_outputs_for(i):
            o = os.path.join(self.get_target_private_dir(t), o_base)
            args = []
            for arg in base_args:
                arg = arg.replace('@INPUT@', infilename)
                arg = arg.replace('@OUTPUT@', o).replace('@BUILD_DIR@', self.get_target_private_dir(t))
                arg = arg.replace('@CURRENT_SOURCE_DIR@', os.path.join(self.build_to_src, t.subdir))
                args.append(arg)
            args = self.replace_outputs(args, self.get_target_private_dir(t), outfilelist)
            args = self.replace_extra_args(args, genlist)
            if generator.capture:
                full_command = ['('] + exe_arr + args + ['>', o, ')']
            else:
                full_command = exe_arr + args
            commands.append(full_command)
    gen_dict.add_item('runOnlyForDeploymentPostprocessing', 0)
    gen_dict.add_item('shellPath', '/bin/sh')
    quoted_cmds = []
    for cmnd in commands:
        q = []
        for c in cmnd:
            if ' ' in c:
                q.append(f'\\"{c}\\"')
            else:
                q.append(c)
        quoted_cmds.append(' '.join(q))
    cmdstr = '"' + ' && '.join(quoted_cmds) + '"'
    gen_dict.add_item('shellScript', cmdstr)
    gen_dict.add_item('showEnvVarsInLog', 0)