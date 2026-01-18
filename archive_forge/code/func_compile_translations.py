from __future__ import annotations
import os
import shutil
import typing as T
import xml.etree.ElementTree as ET
from . import ModuleReturnValue, ExtensionModule
from .. import build
from .. import coredata
from .. import mlog
from ..dependencies import find_external_dependency, Dependency, ExternalLibrary, InternalDependency
from ..mesonlib import MesonException, File, version_compare, Popen_safe
from ..interpreter import extract_required_kwarg
from ..interpreter.type_checking import INSTALL_DIR_KW, INSTALL_KW, NoneType
from ..interpreterbase import ContainerTypeInfo, FeatureDeprecated, KwargInfo, noPosargs, FeatureNew, typed_kwargs
from ..programs import NonExistingExternalProgram
@FeatureNew('qt.compile_translations', '0.44.0')
@noPosargs
@typed_kwargs('qt.compile_translations', KwargInfo('build_by_default', bool, default=False), INSTALL_KW, INSTALL_DIR_KW, KwargInfo('method', str, default='auto'), KwargInfo('qresource', (str, NoneType), since='0.56.0'), KwargInfo('rcc_extra_arguments', ContainerTypeInfo(list, str), listify=True, default=[], since='0.56.0'), KwargInfo('ts_files', ContainerTypeInfo(list, (str, File, build.CustomTarget, build.CustomTargetIndex, build.GeneratedList)), listify=True, default=[]))
def compile_translations(self, state: 'ModuleState', args: T.Tuple, kwargs: 'CompileTranslationsKwArgs') -> ModuleReturnValue:
    ts_files = kwargs['ts_files']
    if any((isinstance(s, (build.CustomTarget, build.CustomTargetIndex, build.GeneratedList)) for s in ts_files)):
        FeatureNew.single_use('qt.compile_translations: custom_target or generator for "ts_files" keyword argument', '0.60.0', state.subproject, location=state.current_node)
    if kwargs['install'] and (not kwargs['install_dir']):
        raise MesonException('qt.compile_translations: "install_dir" keyword argument must be set when "install" is true.')
    qresource = kwargs['qresource']
    if qresource:
        if ts_files:
            raise MesonException('qt.compile_translations: Cannot specify both ts_files and qresource')
        if os.path.dirname(qresource) != '':
            raise MesonException('qt.compile_translations: qresource file name must not contain a subdirectory.')
        qresource_file = File.from_built_file(state.subdir, qresource)
        infile_abs = os.path.join(state.environment.source_dir, qresource_file.relative_name())
        outfile_abs = os.path.join(state.environment.build_dir, qresource_file.relative_name())
        os.makedirs(os.path.dirname(outfile_abs), exist_ok=True)
        shutil.copy2(infile_abs, outfile_abs)
        self.interpreter.add_build_def_file(infile_abs)
        _, nodes = self._qrc_nodes(state, qresource_file)
        for c in nodes:
            if c.endswith('.qm'):
                ts_files.append(c.rstrip('.qm') + '.ts')
            else:
                raise MesonException(f'qt.compile_translations: qresource can only contain qm files, found {c}')
        results = self.preprocess(state, [], {'qresources': qresource_file, 'rcc_extra_arguments': kwargs['rcc_extra_arguments']})
    self._detect_tools(state, kwargs['method'])
    translations: T.List[build.CustomTarget] = []
    for ts in ts_files:
        if not self.tools['lrelease'].found():
            raise MesonException('qt.compile_translations: ' + self.tools['lrelease'].name + ' not found')
        if qresource:
            assert isinstance(ts, str), 'for mypy'
            outdir = os.path.dirname(os.path.normpath(os.path.join(state.subdir, ts)))
            ts = os.path.basename(ts)
        else:
            outdir = state.subdir
        cmd: T.List[T.Union[ExternalProgram, build.Executable, str]] = [self.tools['lrelease'], '@INPUT@', '-qm', '@OUTPUT@']
        lrelease_target = build.CustomTarget(f'qt{self.qt_version}-compile-{ts}', outdir, state.subproject, state.environment, cmd, [ts], ['@BASENAME@.qm'], install=kwargs['install'], install_dir=[kwargs['install_dir']], install_tag=['i18n'], build_by_default=kwargs['build_by_default'], description='Compiling Qt translations {}')
        translations.append(lrelease_target)
    if qresource:
        return ModuleReturnValue(results.return_value[0], [results.new_objects, translations])
    else:
        return ModuleReturnValue(translations, [translations])