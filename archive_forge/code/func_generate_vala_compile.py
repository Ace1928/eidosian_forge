from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from pathlib import PurePath, Path
from textwrap import dedent
import itertools
import json
import os
import pickle
import re
import subprocess
import typing as T
from . import backends
from .. import modules
from .. import environment, mesonlib
from .. import build
from .. import mlog
from .. import compilers
from ..arglist import CompilerArgs
from ..compilers import Compiler
from ..linkers import ArLikeLinker, RSPFileSyntax
from ..mesonlib import (
from ..mesonlib import get_compiler_for_source, has_path_sep, OptionKey
from .backends import CleanTrees
from ..build import GeneratedList, InvalidArguments
def generate_vala_compile(self, target: build.BuildTarget) -> T.Tuple[T.MutableMapping[str, File], T.MutableMapping[str, File], T.List[str]]:
    """Vala is compiled into C. Set up all necessary build steps here."""
    vala_src, vapi_src, other_src = self.split_vala_sources(target)
    extra_dep_files = []
    if not vala_src:
        raise InvalidArguments(f'Vala library {target.name!r} has no Vala or Genie source files.')
    valac = target.compilers['vala']
    c_out_dir = self.get_target_private_dir(target)
    vala_c_src: T.List[str] = []
    valac_outputs: T.List = []
    all_files = list(vapi_src)
    srcbasedir = os.path.join(self.build_to_src, target.get_subdir())
    for vala_file, gensrc in vala_src.items():
        all_files.append(vala_file)
        if isinstance(gensrc, (build.CustomTarget, build.GeneratedList)) or gensrc.is_built:
            vala_c_file = os.path.splitext(os.path.basename(vala_file))[0] + '.c'
            abs_srcbasedir = os.path.join(self.environment.get_source_dir(), target.get_subdir())
            abs_vala_file = os.path.join(self.environment.get_build_dir(), vala_file)
            if PurePath(os.path.commonpath((abs_srcbasedir, abs_vala_file))) == PurePath(abs_srcbasedir):
                vala_c_subdir = PurePath(abs_vala_file).parent.relative_to(abs_srcbasedir)
                vala_c_file = os.path.join(str(vala_c_subdir), vala_c_file)
        else:
            path_to_target = os.path.join(self.build_to_src, target.get_subdir())
            if vala_file.startswith(path_to_target):
                vala_c_file = os.path.splitext(os.path.relpath(vala_file, path_to_target))[0] + '.c'
            else:
                vala_c_file = os.path.splitext(os.path.basename(vala_file))[0] + '.c'
        vala_c_file = os.path.join(c_out_dir, vala_c_file)
        vala_c_src.append(vala_c_file)
        valac_outputs.append(vala_c_file)
    args = self.generate_basic_compiler_args(target, valac)
    args += valac.get_colorout_args(target.get_option(OptionKey('b_colorout')))
    args += ['--directory', c_out_dir]
    args += ['--basedir', srcbasedir]
    if target.is_linkable_target():
        args += ['--library', target.name]
        hname = os.path.join(self.get_target_dir(target), target.vala_header)
        args += ['--header', hname]
        if target.is_unity:
            args += ['--use-header']
        valac_outputs.append(hname)
        vapiname = os.path.join(self.get_target_dir(target), target.vala_vapi)
        args += ['--vapi', os.path.join('..', target.vala_vapi)]
        valac_outputs.append(vapiname)
        if len(target.install_dir) > 1 and target.install_dir[1] is True:
            target.install_dir[1] = self.environment.get_includedir()
        if len(target.install_dir) > 2 and target.install_dir[2] is True:
            target.install_dir[2] = os.path.join(self.environment.get_datadir(), 'vala', 'vapi')
        if isinstance(target.vala_gir, str):
            girname = os.path.join(self.get_target_dir(target), target.vala_gir)
            args += ['--gir', os.path.join('..', target.vala_gir)]
            valac_outputs.append(girname)
            if len(target.install_dir) > 3 and target.install_dir[3] is True:
                target.install_dir[3] = os.path.join(self.environment.get_datadir(), 'gir-1.0')
    gres_dirs = []
    for gensrc in other_src[1].values():
        if isinstance(gensrc, modules.GResourceTarget):
            gres_xml, = self.get_custom_target_sources(gensrc)
            args += ['--gresources=' + gres_xml]
            for source_dir in gensrc.source_dirs:
                gres_dirs += [os.path.join(self.get_target_dir(gensrc), source_dir)]
            gres_c, = gensrc.get_outputs()
            extra_dep_files += [os.path.join(self.get_target_dir(gensrc), gres_c)]
    for gres_dir in OrderedSet(gres_dirs):
        args += [f'--gresourcesdir={gres_dir}']
    dependency_vapis = self.determine_dep_vapis(target)
    extra_dep_files += dependency_vapis
    extra_dep_files.extend(self.get_target_depend_files(target))
    args += target.get_extra_args('vala')
    element = NinjaBuildElement(self.all_outputs, valac_outputs, self.compiler_to_rule_name(valac), all_files + dependency_vapis)
    element.add_item('ARGS', args)
    element.add_dep(extra_dep_files)
    self.add_build(element)
    self.create_target_source_introspection(target, valac, args, all_files, [])
    return (other_src[0], other_src[1], vala_c_src)