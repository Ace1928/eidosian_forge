from __future__ import annotations
import copy, json, os, shutil, re
import typing as T
from . import ExtensionModule, ModuleInfo
from .. import mesonlib
from .. import mlog
from ..coredata import UserFeatureOption
from ..build import known_shmod_kwargs, CustomTarget, CustomTargetIndex, BuildTarget, GeneratedList, StructuredSources, ExtractedObjects, SharedModule
from ..dependencies import NotFoundDependency
from ..dependencies.detect import get_dep_identifier, find_external_dependency
from ..dependencies.python import BasicPythonExternalProgram, python_factory, _PythonDependencyBase
from ..interpreter import extract_required_kwarg, permitted_dependency_kwargs, primitives as P_OBJ
from ..interpreter.interpreterobjects import _ExternalProgramHolder
from ..interpreter.type_checking import NoneType, PRESERVE_PATH_KW, SHARED_MOD_KWS
from ..interpreterbase import (
from ..mesonlib import MachineChoice, OptionKey
from ..programs import ExternalProgram, NonExistingExternalProgram
@permittedKwargs(mod_kwargs)
@typed_pos_args('python.extension_module', str, varargs=(str, mesonlib.File, CustomTarget, CustomTargetIndex, GeneratedList, StructuredSources, ExtractedObjects, BuildTarget))
@typed_kwargs('python.extension_module', *_MOD_KWARGS, _DEFAULTABLE_SUBDIR_KW, _LIMITED_API_KW, allow_unknown=True)
def extension_module_method(self, args: T.Tuple[str, T.List[BuildTargetSource]], kwargs: ExtensionModuleKw) -> 'SharedModule':
    if 'install_dir' in kwargs:
        if kwargs['subdir'] is not None:
            raise InvalidArguments('"subdir" and "install_dir" are mutually exclusive')
    else:
        subdir = kwargs.pop('subdir') or ''
        kwargs['install_dir'] = self._get_install_dir_impl(False, subdir)
    target_suffix = self.suffix
    new_deps = mesonlib.extract_as_list(kwargs, 'dependencies')
    pydep = next((dep for dep in new_deps if isinstance(dep, _PythonDependencyBase)), None)
    if pydep is None:
        pydep = self._dependency_method_impl({})
        if not pydep.found():
            raise mesonlib.MesonException('Python dependency not found')
        new_deps.append(pydep)
        FeatureNew.single_use('python_installation.extension_module with implicit dependency on python', '0.63.0', self.subproject, 'use python_installation.dependency()', self.current_node)
    limited_api_version = kwargs.pop('limited_api')
    allow_limited_api = self.interpreter.environment.coredata.get_option(OptionKey('allow_limited_api', module='python'))
    if limited_api_version != '' and allow_limited_api:
        target_suffix = self.limited_api_suffix
        limited_api_version_hex = self._convert_api_version_to_py_version_hex(limited_api_version, pydep.version)
        limited_api_definition = f'-DPy_LIMITED_API={limited_api_version_hex}'
        new_c_args = mesonlib.extract_as_list(kwargs, 'c_args')
        new_c_args.append(limited_api_definition)
        kwargs['c_args'] = new_c_args
        new_cpp_args = mesonlib.extract_as_list(kwargs, 'cpp_args')
        new_cpp_args.append(limited_api_definition)
        kwargs['cpp_args'] = new_cpp_args
        for_machine = kwargs['native']
        compilers = self.interpreter.environment.coredata.compilers[for_machine]
        if any((compiler.get_id() == 'msvc' for compiler in compilers.values())):
            pydep_copy = copy.copy(pydep)
            pydep_copy.find_libpy_windows(self.env, limited_api=True)
            if not pydep_copy.found():
                raise mesonlib.MesonException('Python dependency supporting limited API not found')
            new_deps.remove(pydep)
            new_deps.append(pydep_copy)
            pyver = pydep.version.replace('.', '')
            python_windows_debug_link_exception = f'/NODEFAULTLIB:python{pyver}_d.lib'
            python_windows_release_link_exception = f'/NODEFAULTLIB:python{pyver}.lib'
            new_link_args = mesonlib.extract_as_list(kwargs, 'link_args')
            is_debug = self.interpreter.environment.coredata.options[OptionKey('debug')].value
            if is_debug:
                new_link_args.append(python_windows_debug_link_exception)
            else:
                new_link_args.append(python_windows_release_link_exception)
            kwargs['link_args'] = new_link_args
    kwargs['dependencies'] = new_deps
    split, target_suffix = target_suffix.rsplit('.', 1)
    args = (args[0] + split, args[1])
    kwargs['name_prefix'] = ''
    kwargs['name_suffix'] = target_suffix
    if kwargs['gnu_symbol_visibility'] == '' and (self.is_pypy or mesonlib.version_compare(self.version, '>=3.9')):
        kwargs['gnu_symbol_visibility'] = 'inlineshidden'
    return self.interpreter.build_target(self.current_node, args, kwargs, SharedModule)