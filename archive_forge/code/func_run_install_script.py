from __future__ import annotations
from glob import glob
import argparse
import errno
import os
import selectors
import shlex
import shutil
import subprocess
import sys
import typing as T
import re
from . import build, environment
from .backend.backends import InstallData
from .mesonlib import (MesonException, Popen_safe, RealPathAction, is_windows,
from .scripts import depfixer, destdir_join
from .scripts.meson_exe import run_exe
def run_install_script(self, d: InstallData, destdir: str, fullprefix: str) -> None:
    env = {'MESON_SOURCE_ROOT': d.source_dir, 'MESON_BUILD_ROOT': d.build_dir, 'MESONINTROSPECT': ' '.join([shlex.quote(x) for x in d.mesonintrospect])}
    if self.options.quiet:
        env['MESON_INSTALL_QUIET'] = '1'
    if self.dry_run:
        env['MESON_INSTALL_DRY_RUN'] = '1'
    for i in d.install_scripts:
        if not self.should_install(i):
            continue
        if i.installdir_map is not None:
            mapp = i.installdir_map
        else:
            mapp = {'prefix': d.prefix}
        localenv = env.copy()
        localenv.update({'MESON_INSTALL_' + k.upper(): os.path.join(d.prefix, v) for k, v in mapp.items()})
        localenv.update({'MESON_INSTALL_DESTDIR_' + k.upper(): get_destdir_path(destdir, fullprefix, v) for k, v in mapp.items()})
        name = ' '.join(i.cmd_args)
        if i.skip_if_destdir and destdir:
            self.log(f'Skipping custom install script because DESTDIR is set {name!r}')
            continue
        self.did_install_something = True
        self.log(f'Running custom install script {name!r}')
        try:
            rc = self.run_exe(i, localenv)
        except OSError:
            print(f"FAILED: install script '{name}' could not be run.")
            sys.exit(127)
        if rc != 0:
            print(f"FAILED: install script '{name}' failed with exit code {rc}.")
            sys.exit(rc)