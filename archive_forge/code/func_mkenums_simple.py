from __future__ import annotations
import copy
import itertools
import functools
import os
import subprocess
import textwrap
import typing as T
from . import (
from .. import build
from .. import interpreter
from .. import mesonlib
from .. import mlog
from ..build import CustomTarget, CustomTargetIndex, Executable, GeneratedList, InvalidArguments
from ..dependencies import Dependency, InternalDependency
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import DEPENDS_KW, DEPEND_FILES_KW, ENV_KW, INSTALL_DIR_KW, INSTALL_KW, NoneType, DEPENDENCY_SOURCES_KW, in_set_validator
from ..interpreterbase import noPosargs, noKwargs, FeatureNew, FeatureDeprecated
from ..interpreterbase import typed_kwargs, KwargInfo, ContainerTypeInfo
from ..interpreterbase.decorators import typed_pos_args
from ..mesonlib import (
from ..programs import OverrideProgram
from ..scripts.gettext import read_linguas
@FeatureNew('gnome.mkenums_simple', '0.42.0')
@typed_pos_args('gnome.mkenums_simple', str)
@typed_kwargs('gnome.mkenums_simple', *_MK_ENUMS_COMMON_KWS, KwargInfo('sources', ContainerTypeInfo(list, (str, mesonlib.File)), listify=True, required=True), KwargInfo('header_prefix', str, default=''), KwargInfo('function_prefix', str, default=''), KwargInfo('body_prefix', str, default=''), KwargInfo('decorator', str, default=''))
def mkenums_simple(self, state: 'ModuleState', args: T.Tuple[str], kwargs: 'MkEnumsSimple') -> ModuleReturnValue:
    hdr_filename = f'{args[0]}.h'
    body_filename = f'{args[0]}.c'
    header_prefix = kwargs['header_prefix']
    decl_decorator = kwargs['decorator']
    func_prefix = kwargs['function_prefix']
    body_prefix = kwargs['body_prefix']
    cmd: T.List[str] = []
    if kwargs['identifier_prefix']:
        cmd.extend(['--identifier-prefix', kwargs['identifier_prefix']])
    if kwargs['symbol_prefix']:
        cmd.extend(['--symbol-prefix', kwargs['symbol_prefix']])
    c_cmd = cmd.copy()
    fhead = ''
    if body_prefix != '':
        fhead += '%s\n' % body_prefix
    fhead += '#include "%s"\n' % hdr_filename
    for hdr in self.interpreter.source_strings_to_files(kwargs['sources']):
        hdr_path = os.path.relpath(hdr.relative_name(), state.subdir)
        fhead += f'#include "{hdr_path}"\n'
    fhead += textwrap.dedent('\n            #define C_ENUM(v) ((gint) v)\n            #define C_FLAGS(v) ((guint) v)\n            ')
    c_cmd.extend(['--fhead', fhead])
    c_cmd.append('--fprod')
    c_cmd.append(textwrap.dedent('\n            /* enumerations from "@basename@" */\n            '))
    c_cmd.append('--vhead')
    c_cmd.append(textwrap.dedent(f'\n            GType\n            {func_prefix}@enum_name@_get_type (void)\n            {{\n            static gsize gtype_id = 0;\n            static const G@Type@Value values[] = {{'))
    c_cmd.extend(['--vprod', '    { C_@TYPE@(@VALUENAME@), "@VALUENAME@", "@valuenick@" },'])
    c_cmd.append('--vtail')
    c_cmd.append(textwrap.dedent('    { 0, NULL, NULL }\n            };\n            if (g_once_init_enter (&gtype_id)) {\n                GType new_type = g_@type@_register_static (g_intern_static_string ("@EnumName@"), values);\n                g_once_init_leave (&gtype_id, new_type);\n            }\n            return (GType) gtype_id;\n            }'))
    c_cmd.append('@INPUT@')
    c_file = self._make_mkenum_impl(state, kwargs['sources'], body_filename, c_cmd)
    h_cmd = cmd.copy()
    h_cmd.append('--fhead')
    h_cmd.append(textwrap.dedent(f'#pragma once\n\n            #include <glib-object.h>\n            {header_prefix}\n\n            G_BEGIN_DECLS\n            '))
    h_cmd.append('--fprod')
    h_cmd.append(textwrap.dedent('\n            /* enumerations from "@basename@" */\n            '))
    h_cmd.append('--vhead')
    h_cmd.append(textwrap.dedent(f'\n            {decl_decorator}\n            GType {func_prefix}@enum_name@_get_type (void);\n            #define @ENUMPREFIX@_TYPE_@ENUMSHORT@ ({func_prefix}@enum_name@_get_type())'))
    h_cmd.append('--ftail')
    h_cmd.append(textwrap.dedent('\n            G_END_DECLS'))
    h_cmd.append('@INPUT@')
    h_file = self._make_mkenum_impl(state, kwargs['sources'], hdr_filename, h_cmd, install=kwargs['install_header'], install_dir=kwargs['install_dir'])
    return ModuleReturnValue([c_file, h_file], [c_file, h_file])