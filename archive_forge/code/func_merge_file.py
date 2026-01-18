from __future__ import annotations
from os import path
import shlex
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import build
from .. import mesonlib
from .. import mlog
from ..interpreter.type_checking import CT_BUILD_BY_DEFAULT, CT_INPUT_KW, INSTALL_TAG_KW, OUTPUT_KW, INSTALL_DIR_KW, INSTALL_KW, NoneType, in_set_validator
from ..interpreterbase import FeatureNew, InvalidArguments
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, noPosargs, typed_kwargs, typed_pos_args
from ..programs import ExternalProgram
from ..scripts.gettext import read_linguas
@FeatureNew('i18n.merge_file', '0.37.0')
@noPosargs
@typed_kwargs('i18n.merge_file', CT_BUILD_BY_DEFAULT, CT_INPUT_KW, KwargInfo('install_dir', (str, NoneType)), INSTALL_TAG_KW, OUTPUT_KW, INSTALL_KW, _ARGS.evolve(since='0.51.0'), _DATA_DIRS.evolve(since='0.41.0'), KwargInfo('po_dir', str, required=True), KwargInfo('type', str, default='xml', validator=in_set_validator({'xml', 'desktop'})))
def merge_file(self, state: 'ModuleState', args: T.List['TYPE_var'], kwargs: 'MergeFile') -> ModuleReturnValue:
    if kwargs['install'] and (not kwargs['install_dir']):
        raise InvalidArguments('i18n.merge_file: "install_dir" keyword argument must be set when "install" is true.')
    if self.tools['msgfmt'] is None or not self.tools['msgfmt'].found():
        self.tools['msgfmt'] = state.find_program('msgfmt', for_machine=mesonlib.MachineChoice.BUILD)
    if isinstance(self.tools['msgfmt'], ExternalProgram):
        try:
            have_version = self.tools['msgfmt'].get_version()
        except mesonlib.MesonException as e:
            raise mesonlib.MesonException('i18n.merge_file requires GNU msgfmt') from e
        want_version = '>=0.19' if kwargs['type'] == 'desktop' else '>=0.19.7'
        if not mesonlib.version_compare(have_version, want_version):
            msg = f'i18n.merge_file requires GNU msgfmt {want_version} to produce files of type: ' + kwargs['type'] + f' (got: {have_version})'
            raise mesonlib.MesonException(msg)
    podir = path.join(state.build_to_src, state.subdir, kwargs['po_dir'])
    ddirs = self._get_data_dirs(state, kwargs['data_dirs'])
    datadirs = '--datadirs=' + ':'.join(ddirs) if ddirs else None
    command: T.List[T.Union[str, build.BuildTarget, build.CustomTarget, build.CustomTargetIndex, 'ExternalProgram', mesonlib.File]] = []
    command.extend(state.environment.get_build_command())
    command.extend(['--internal', 'msgfmthelper', '--msgfmt=' + self.tools['msgfmt'].get_path()])
    if datadirs:
        command.append(datadirs)
    command.extend(['@INPUT@', '@OUTPUT@', kwargs['type'], podir])
    if kwargs['args']:
        command.append('--')
        command.extend(kwargs['args'])
    build_by_default = kwargs['build_by_default']
    if build_by_default is None:
        build_by_default = kwargs['install']
    install_tag = [kwargs['install_tag']] if kwargs['install_tag'] is not None else None
    ct = build.CustomTarget('', state.subdir, state.subproject, state.environment, command, kwargs['input'], [kwargs['output']], build_by_default=build_by_default, install=kwargs['install'], install_dir=[kwargs['install_dir']] if kwargs['install_dir'] is not None else None, install_tag=install_tag, description='Merging translations for {}')
    return ModuleReturnValue(ct, [ct])