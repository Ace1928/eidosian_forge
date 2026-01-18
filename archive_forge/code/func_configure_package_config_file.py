from __future__ import annotations
import re
import os, os.path, pathlib
import shutil
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleObject, ModuleInfo
from .. import build, mesonlib, mlog, dependencies
from ..cmake import TargetOptions, cmake_defines_to_args
from ..interpreter import SubprojectHolder
from ..interpreter.type_checking import REQUIRED_KW, INSTALL_DIR_KW, NoneType, in_set_validator
from ..interpreterbase import (
@noPosargs
@typed_kwargs('cmake.configure_package_config_file', KwargInfo('configuration', (build.ConfigurationData, dict), required=True), KwargInfo('input', (str, mesonlib.File, ContainerTypeInfo(list, mesonlib.File)), required=True, validator=lambda x: 'requires exactly one file' if isinstance(x, list) and len(x) != 1 else None, convertor=lambda x: x[0] if isinstance(x, list) else x), KwargInfo('name', str, required=True), INSTALL_DIR_KW)
def configure_package_config_file(self, state: ModuleState, args: TYPE_var, kwargs: 'ConfigurePackageConfigFile') -> build.Data:
    inputfile = kwargs['input']
    if isinstance(inputfile, str):
        inputfile = mesonlib.File.from_source_file(state.environment.source_dir, state.subdir, inputfile)
    ifile_abs = inputfile.absolute_path(state.environment.source_dir, state.environment.build_dir)
    name = kwargs['name']
    ofile_path, ofile_fname = os.path.split(os.path.join(state.subdir, f'{name}Config.cmake'))
    ofile_abs = os.path.join(state.environment.build_dir, ofile_path, ofile_fname)
    install_dir = kwargs['install_dir']
    if install_dir is None:
        install_dir = os.path.join(state.environment.coredata.get_option(mesonlib.OptionKey('libdir')), 'cmake', name)
    conf = kwargs['configuration']
    if isinstance(conf, dict):
        FeatureNew.single_use('cmake.configure_package_config_file dict as configuration', '0.62.0', state.subproject, location=state.current_node)
        conf = build.ConfigurationData(conf)
    prefix = state.environment.coredata.get_option(mesonlib.OptionKey('prefix'))
    abs_install_dir = install_dir
    if not os.path.isabs(abs_install_dir):
        abs_install_dir = os.path.join(prefix, install_dir)
    PACKAGE_RELATIVE_PATH = pathlib.PurePath(os.path.relpath(prefix, abs_install_dir)).as_posix()
    extra = ''
    if re.match('^(/usr)?/lib(64)?/.+', abs_install_dir):
        extra = PACKAGE_INIT_EXT.replace('@absInstallDir@', abs_install_dir)
        extra = extra.replace('@installPrefix@', prefix)
    self.create_package_file(ifile_abs, ofile_abs, PACKAGE_RELATIVE_PATH, extra, conf)
    conf.used = True
    conffile = os.path.normpath(inputfile.relative_name())
    self.interpreter.build_def_files.add(conffile)
    res = build.Data([mesonlib.File(True, ofile_path, ofile_fname)], install_dir, install_dir, None, state.subproject)
    self.interpreter.build.data.append(res)
    return res