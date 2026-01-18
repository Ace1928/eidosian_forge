from __future__ import annotations
import copy
import itertools
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
import typing as T
from pathlib import Path, PurePath, PureWindowsPath
import re
from collections import Counter
from . import backends
from .. import build
from .. import mlog
from .. import compilers
from .. import mesonlib
from ..mesonlib import (
from ..environment import Environment, build_filename
from .. import coredata
def get_args_defines_and_inc_dirs(self, target, compiler, generated_files_include_dirs, proj_to_src_root, proj_to_src_dir, build_args):
    target_args = []
    target_defines = []
    target_inc_dirs = []
    file_args: T.Dict[str, CompilerArgs] = {l: c.compiler_args() for l, c in target.compilers.items()}
    file_defines = {l: [] for l in target.compilers}
    file_inc_dirs = {l: [] for l in target.compilers}
    for l, comp in target.compilers.items():
        if l in file_args:
            file_args[l] += compilers.get_base_compile_args(target.get_options(), comp)
            file_args[l] += comp.get_option_compile_args(target.get_options())
    for l, args in self.build.projects_args[target.for_machine].get(target.subproject, {}).items():
        if l in file_args:
            file_args[l] += args
    for l, args in self.build.global_args[target.for_machine].items():
        if l in file_args:
            file_args[l] += args
    for l in file_args.keys():
        file_args[l] += target.get_option(OptionKey('args', machine=target.for_machine, lang=l))
    for args in file_args.values():
        args += ['%(AdditionalOptions)', '%(PreprocessorDefinitions)', '%(AdditionalIncludeDirectories)']
        args += ['-I' + arg for arg in generated_files_include_dirs]
        for d in reversed(target.get_include_dirs()):
            for i in reversed(d.get_incdirs()):
                curdir = os.path.join(d.get_curdir(), i)
                try:
                    args.append('-I' + os.path.join(proj_to_src_root, curdir))
                    args.append('-I' + self.relpath(curdir, target.subdir))
                except ValueError:
                    args.append('-I' + os.path.normpath(curdir))
            for i in d.get_extra_build_dirs():
                curdir = os.path.join(d.get_curdir(), i)
                args.append('-I' + self.relpath(curdir, target.subdir))
    for l, args in target.extra_args.items():
        if l in file_args:
            file_args[l] += args
    for args in file_args.values():
        t_inc_dirs = [self.relpath(self.get_target_private_dir(target), self.get_target_dir(target))]
        if target.implicit_include_directories:
            t_inc_dirs += ['.', proj_to_src_dir]
        args += ['-I' + arg for arg in t_inc_dirs]
    for l, args in file_args.items():
        for arg in args[:]:
            if arg.startswith(('-D', '/D')) or arg == '%(PreprocessorDefinitions)':
                file_args[l].remove(arg)
                if arg == '%(PreprocessorDefinitions)':
                    define = arg
                else:
                    define = arg[2:]
                if define not in file_defines[l]:
                    file_defines[l].append(define)
            elif arg.startswith(('-I', '/I')) or arg == '%(AdditionalIncludeDirectories)':
                file_args[l].remove(arg)
                if arg == '%(AdditionalIncludeDirectories)':
                    inc_dir = arg
                else:
                    inc_dir = arg[2:]
                if inc_dir not in file_inc_dirs[l]:
                    file_inc_dirs[l].append(inc_dir)
                if inc_dir not in target_inc_dirs:
                    target_inc_dirs.append(inc_dir)
    for d in reversed(target.get_external_deps()):
        if d.name != 'openmp':
            d_compile_args = compiler.unix_args_to_native(d.get_compile_args())
            for arg in d_compile_args:
                if arg.startswith(('-D', '/D')):
                    define = arg[2:]
                    if define in target_defines:
                        target_defines.remove(define)
                    target_defines.append(define)
                elif arg.startswith(('-I', '/I')):
                    inc_dir = arg[2:]
                    if inc_dir not in target_inc_dirs:
                        target_inc_dirs.append(inc_dir)
                else:
                    target_args.append(arg)
    if '/Gw' in build_args:
        target_args.append('/Gw')
    return ((target_args, file_args), (target_defines, file_defines), (target_inc_dirs, file_inc_dirs))