from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
def generate_target_install(self, d: InstallData) -> None:
    for t in self.build.get_targets().values():
        if not t.should_install():
            continue
        outdirs, install_dir_names, custom_install_dir = t.get_install_dir()
        num_outdirs, num_out = (len(outdirs), len(t.get_outputs()))
        if num_outdirs not in {1, num_out}:
            m = 'Target {!r} has {} outputs: {!r}, but only {} "install_dir"s were found.\nPass \'false\' for outputs that should not be installed and \'true\' for\nusing the default installation directory for an output.'
            raise MesonException(m.format(t.name, num_out, t.get_outputs(), num_outdirs))
        assert len(t.install_tag) == num_out
        install_mode = t.get_custom_install_mode()
        first_outdir = outdirs[0]
        first_outdir_name = install_dir_names[0]
        if isinstance(t, build.BuildTarget):
            can_strip = not isinstance(t, build.StaticLibrary)
            should_strip = can_strip and t.get_option(OptionKey('strip'))
            assert isinstance(should_strip, bool), 'for mypy'
            if first_outdir is not False:
                tag = t.install_tag[0] or ('devel' if isinstance(t, build.StaticLibrary) else 'runtime')
                mappings = t.get_link_deps_mapping(d.prefix)
                i = TargetInstallData(self.get_target_filename(t), first_outdir, first_outdir_name, should_strip, mappings, t.rpath_dirs_to_remove, t.install_rpath, install_mode, t.subproject, tag=tag, can_strip=can_strip)
                d.targets.append(i)
                for alias, to, tag in t.get_aliases():
                    alias = os.path.join(first_outdir, alias)
                    s = InstallSymlinkData(to, alias, first_outdir, t.subproject, tag, allow_missing=True)
                    d.symlinks.append(s)
                if isinstance(t, (build.SharedLibrary, build.SharedModule, build.Executable)):
                    if t.get_import_filename():
                        if custom_install_dir:
                            implib_install_dir = first_outdir
                        else:
                            implib_install_dir = self.environment.get_import_lib_dir()
                        i = TargetInstallData(self.get_target_filename_for_linking(t), implib_install_dir, first_outdir_name, False, {}, set(), '', install_mode, t.subproject, optional=isinstance(t, build.SharedModule), tag='devel')
                        d.targets.append(i)
                    if not should_strip and t.get_debug_filename():
                        debug_file = os.path.join(self.get_target_dir(t), t.get_debug_filename())
                        i = TargetInstallData(debug_file, first_outdir, first_outdir_name, False, {}, set(), '', install_mode, t.subproject, optional=True, tag='devel')
                        d.targets.append(i)
            if num_outdirs > 1:
                for output, outdir, outdir_name, tag in zip(t.get_outputs()[1:], outdirs[1:], install_dir_names[1:], t.install_tag[1:]):
                    if outdir is False:
                        continue
                    f = os.path.join(self.get_target_dir(t), output)
                    i = TargetInstallData(f, outdir, outdir_name, False, {}, set(), None, install_mode, t.subproject, tag=tag)
                    d.targets.append(i)
        elif isinstance(t, build.CustomTarget):
            if num_outdirs == 1 and num_out > 1:
                if first_outdir is not False:
                    for output, tag in zip(t.get_outputs(), t.install_tag):
                        tag = tag or self.guess_install_tag(output, first_outdir)
                        f = os.path.join(self.get_target_dir(t), output)
                        i = TargetInstallData(f, first_outdir, first_outdir_name, False, {}, set(), None, install_mode, t.subproject, optional=not t.build_by_default, tag=tag)
                        d.targets.append(i)
            else:
                for output, outdir, outdir_name, tag in zip(t.get_outputs(), outdirs, install_dir_names, t.install_tag):
                    if outdir is False:
                        continue
                    tag = tag or self.guess_install_tag(output, outdir)
                    f = os.path.join(self.get_target_dir(t), output)
                    i = TargetInstallData(f, outdir, outdir_name, False, {}, set(), None, install_mode, t.subproject, optional=not t.build_by_default, tag=tag)
                    d.targets.append(i)