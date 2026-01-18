from __future__ import annotations
import itertools
import os
import typing as T
from mesonbuild.interpreterbase.decorators import FeatureNew
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import mesonlib, mlog
from ..build import (BothLibraries, BuildTarget, CustomTargetIndex, Executable, ExtractedObjects, GeneratedList,
from ..compilers.compilers import are_asserts_disabled, lang_suffixes
from ..interpreter.type_checking import (
from ..interpreterbase import ContainerTypeInfo, InterpreterException, KwargInfo, typed_kwargs, typed_pos_args, noPosargs, permittedKwargs
from ..mesonlib import File
from ..programs import ExternalProgram
@noPosargs
@typed_kwargs('rust.bindgen', KwargInfo('c_args', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('args', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('input', ContainerTypeInfo(list, (File, GeneratedList, BuildTarget, BothLibraries, ExtractedObjects, CustomTargetIndex, CustomTarget, str), allow_empty=False), default=[], listify=True, required=True), KwargInfo('language', (str, NoneType), since='1.4.0', validator=in_set_validator({'c', 'cpp'})), KwargInfo('bindgen_version', ContainerTypeInfo(list, str), default=[], listify=True, since='1.4.0'), INCLUDE_DIRECTORIES.evolve(since_values={ContainerTypeInfo(list, str): '1.0.0'}), OUTPUT_KW, KwargInfo('output_inline_wrapper', str, default='', since='1.4.0'), DEPENDENCIES_KW.evolve(since='1.0.0'))
def bindgen(self, state: ModuleState, args: T.List, kwargs: FuncBindgen) -> ModuleReturnValue:
    """Wrapper around bindgen to simplify it's use.

        The main thing this simplifies is the use of `include_directory`
        objects, instead of having to pass a plethora of `-I` arguments.
        """
    header, *_deps = self.interpreter.source_strings_to_files(kwargs['input'])
    depends: T.List[SourceOutputs] = []
    depend_files: T.List[File] = []
    for d in _deps:
        if isinstance(d, File):
            depend_files.append(d)
        else:
            depends.append(d)
    clang_args = state.environment.properties.host.get_bindgen_clang_args().copy()
    for i in state.process_include_dirs(kwargs['include_directories']):
        clang_args.extend([f'-I{x}' for x in i.to_string_list(state.environment.get_source_dir(), state.environment.get_build_dir())])
    if are_asserts_disabled(state.environment.coredata.options):
        clang_args.append('-DNDEBUG')
    for de in kwargs['dependencies']:
        for i in de.get_include_dirs():
            clang_args.extend([f'-I{x}' for x in i.to_string_list(state.environment.get_source_dir(), state.environment.get_build_dir())])
        clang_args.extend(de.get_all_compile_args())
        for s in de.get_sources():
            if isinstance(s, File):
                depend_files.append(s)
            elif isinstance(s, CustomTarget):
                depends.append(s)
    if self._bindgen_bin is None:
        self._bindgen_bin = state.find_program('bindgen', wanted=kwargs['bindgen_version'])
    name: str
    if isinstance(header, File):
        name = header.fname
    elif isinstance(header, (BuildTarget, BothLibraries, ExtractedObjects, StructuredSources)):
        raise InterpreterException('bindgen source file must be a C header, not an object or build target')
    else:
        name = header.get_outputs()[0]
    language = kwargs['language']
    if language is None:
        ext = os.path.splitext(name)[1][1:]
        if ext in lang_suffixes['cpp']:
            language = 'cpp'
        elif ext == 'h':
            language = 'c'
        else:
            raise InterpreterException(f'Unknown file type extension for: {name}')
    cargs = state.get_option('args', state.subproject, lang=language)
    assert isinstance(cargs, list), 'for mypy'
    for a in itertools.chain(state.global_args.get(language, []), state.project_args.get(language, []), cargs):
        if a.startswith(('-I', '/I', '-D', '/D', '-U', '/U')):
            clang_args.append(a)
    if language == 'cpp':
        clang_args.extend(['-x', 'c++'])
    std = state.get_option('std', lang=language)
    assert isinstance(std, str), 'for mypy'
    if std.startswith('vc++'):
        if std.endswith('latest'):
            mlog.warning('Attempting to translate vc++latest into a clang compatible version.', 'Currently this is hardcoded for c++20', once=True, fatal=False)
            std = 'c++20'
        else:
            mlog.debug('The current C++ standard is a Visual Studio extension version.', 'bindgen will use a the nearest C++ standard instead')
            std = std[1:]
    if std != 'none':
        clang_args.append(f'-std={std}')
    inline_wrapper_args: T.List[str] = []
    outputs = [kwargs['output']]
    if kwargs['output_inline_wrapper']:
        if isinstance(self._bindgen_bin, ExternalProgram):
            if mesonlib.version_compare(self._bindgen_bin.get_version(), '< 0.65'):
                raise InterpreterException("'output_inline_wrapper' parameter of rust.bindgen requires bindgen-0.65 or newer")
        outputs.append(kwargs['output_inline_wrapper'])
        inline_wrapper_args = ['--experimental', '--wrap-static-fns', '--wrap-static-fns-path', os.path.join(state.environment.build_dir, '@OUTPUT1@')]
    cmd = self._bindgen_bin.get_command() + ['@INPUT@', '--output', os.path.join(state.environment.build_dir, '@OUTPUT0@')] + kwargs['args'] + inline_wrapper_args + ['--'] + kwargs['c_args'] + clang_args + ['-MD', '-MQ', '@INPUT@', '-MF', '@DEPFILE@']
    target = CustomTarget(f'rustmod-bindgen-{name}'.replace('/', '_'), state.subdir, state.subproject, state.environment, cmd, [header], outputs, depfile='@PLAINNAME@.d', extra_depends=depends, depend_files=depend_files, backend=state.backend, description='Generating bindings for Rust {}')
    return ModuleReturnValue(target, [target])