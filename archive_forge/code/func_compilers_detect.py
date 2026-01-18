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
def compilers_detect(self, state: 'ModuleState', qt_dep: 'QtDependencyType') -> None:
    """Detect Qt (4 or 5) moc, uic, rcc in the specified bindir or in PATH"""
    wanted = f'== {qt_dep.version}'

    def gen_bins() -> T.Generator[T.Tuple[str, str], None, None]:
        for b in self.tools:
            if qt_dep.bindir:
                yield (os.path.join(qt_dep.bindir, b), b)
            if qt_dep.libexecdir:
                yield (os.path.join(qt_dep.libexecdir, b), b)
            yield (f'{b}{qt_dep.qtver}', b)
            yield (f'{b}-qt{qt_dep.qtver}', b)
            yield (b, b)
    for b, name in gen_bins():
        if self.tools[name].found():
            continue
        if name == 'lrelease':
            arg = ['-version']
        elif version_compare(qt_dep.version, '>= 5'):
            arg = ['--version']
        else:
            arg = ['-v']

        def get_version(p: T.Union[ExternalProgram, build.Executable]) -> str:
            _, out, err = Popen_safe(p.get_command() + arg)
            if name == 'lrelease' or not qt_dep.version.startswith('4'):
                care = out
            else:
                care = err
            return care.rsplit(' ', maxsplit=1)[-1].replace(')', '').strip()
        p = state.find_program(b, required=False, version_func=get_version, wanted=wanted)
        if p.found():
            self.tools[name] = p