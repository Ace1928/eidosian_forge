from __future__ import annotations
import itertools
import shutil
import os
import textwrap
import typing as T
import collections
from . import build
from . import coredata
from . import environment
from . import mesonlib
from . import mintro
from . import mlog
from .ast import AstIDGenerator, IntrospectionInterpreter
from .mesonlib import MachineChoice, OptionKey
def print_conf(self, pager: bool) -> None:
    if pager:
        mlog.start_pager()

    def print_default_values_warning() -> None:
        mlog.warning('The source directory instead of the build directory was specified.')
        mlog.warning('Only the default values for the project are printed.')
    if self.default_values_only:
        print_default_values_warning()
        mlog.log('')
    mlog.log('Core properties:')
    mlog.log('  Source dir', self.source_dir)
    if not self.default_values_only:
        mlog.log('  Build dir ', self.build_dir)
    dir_option_names = set(coredata.BUILTIN_DIR_OPTIONS)
    test_option_names = {OptionKey('errorlogs'), OptionKey('stdsplit')}
    dir_options: 'coredata.MutableKeyedOptionDictType' = {}
    test_options: 'coredata.MutableKeyedOptionDictType' = {}
    core_options: 'coredata.MutableKeyedOptionDictType' = {}
    module_options: T.Dict[str, 'coredata.MutableKeyedOptionDictType'] = collections.defaultdict(dict)
    for k, v in self.coredata.options.items():
        if k in dir_option_names:
            dir_options[k] = v
        elif k in test_option_names:
            test_options[k] = v
        elif k.module:
            if self.build and k.module not in self.build.modules:
                continue
            module_options[k.module][k] = v
        elif k.is_builtin():
            core_options[k] = v
    host_core_options = self.split_options_per_subproject({k: v for k, v in core_options.items() if k.machine is MachineChoice.HOST})
    build_core_options = self.split_options_per_subproject({k: v for k, v in core_options.items() if k.machine is MachineChoice.BUILD})
    host_compiler_options = self.split_options_per_subproject({k: v for k, v in self.coredata.options.items() if k.is_compiler() and k.machine is MachineChoice.HOST})
    build_compiler_options = self.split_options_per_subproject({k: v for k, v in self.coredata.options.items() if k.is_compiler() and k.machine is MachineChoice.BUILD})
    project_options = self.split_options_per_subproject({k: v for k, v in self.coredata.options.items() if k.is_project()})
    show_build_options = self.default_values_only or self.build.environment.is_cross_build()
    self.add_section('Main project options')
    self.print_options('Core options', host_core_options[''])
    if show_build_options:
        self.print_options('', build_core_options[''])
    self.print_options('Backend options', {k: v for k, v in self.coredata.options.items() if k.is_backend()})
    self.print_options('Base options', {k: v for k, v in self.coredata.options.items() if k.is_base()})
    self.print_options('Compiler options', host_compiler_options.get('', {}))
    if show_build_options:
        self.print_options('', build_compiler_options.get('', {}))
    for mod, mod_options in module_options.items():
        self.print_options(f'{mod} module options', mod_options)
    self.print_options('Directories', dir_options)
    self.print_options('Testing options', test_options)
    self.print_options('Project options', project_options.get('', {}))
    for subproject in sorted(self.all_subprojects):
        if subproject == '':
            continue
        self.add_section('Subproject ' + subproject)
        if subproject in host_core_options:
            self.print_options('Core options', host_core_options[subproject])
        if subproject in build_core_options and show_build_options:
            self.print_options('', build_core_options[subproject])
        if subproject in host_compiler_options:
            self.print_options('Compiler options', host_compiler_options[subproject])
        if subproject in build_compiler_options and show_build_options:
            self.print_options('', build_compiler_options[subproject])
        if subproject in project_options:
            self.print_options('Project options', project_options[subproject])
    self.print_aligned()
    if self.default_values_only:
        mlog.log('')
        print_default_values_warning()
    self.print_nondefault_buildtype_options()