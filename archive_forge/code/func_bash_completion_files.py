from __future__ import annotations
import os, subprocess
import argparse
import tempfile
import shutil
import itertools
import typing as T
from pathlib import Path
from . import build, minstall
from .mesonlib import (EnvironmentVariables, MesonException, is_windows, setup_vsenv, OptionKey,
from . import mlog
def bash_completion_files(b: build.Build, install_data: 'InstallData') -> T.List[str]:
    from .dependencies.pkgconfig import PkgConfigDependency
    result = []
    dep = PkgConfigDependency('bash-completion', b.environment, {'required': False, 'silent': True, 'version': '>=2.10'})
    if dep.found():
        prefix = b.environment.coredata.get_option(OptionKey('prefix'))
        assert isinstance(prefix, str), 'for mypy'
        datadir = b.environment.coredata.get_option(OptionKey('datadir'))
        assert isinstance(datadir, str), 'for mypy'
        datadir_abs = os.path.join(prefix, datadir)
        completionsdir = dep.get_variable(pkgconfig='completionsdir', pkgconfig_define=(('datadir', datadir_abs),))
        assert isinstance(completionsdir, str), 'for mypy'
        completionsdir_path = Path(completionsdir)
        for f in install_data.data:
            if completionsdir_path in Path(f.install_path).parents:
                result.append(f.path)
    return result