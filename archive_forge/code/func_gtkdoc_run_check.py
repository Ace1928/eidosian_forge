from __future__ import annotations
import sys, os
import subprocess
import shutil
import argparse
from ..mesonlib import MesonException, Popen_safe, is_windows, is_cygwin, split_args
from . import destdir_join
import typing as T
def gtkdoc_run_check(cmd: T.List[str], cwd: str, library_paths: T.Optional[T.List[str]]=None) -> None:
    if library_paths is None:
        library_paths = []
    env = dict(os.environ)
    if is_windows() or is_cygwin():
        if 'PATH' in env:
            library_paths.extend(env['PATH'].split(os.pathsep))
        env['PATH'] = os.pathsep.join(library_paths)
    else:
        if 'LD_LIBRARY_PATH' in env:
            library_paths.extend(env['LD_LIBRARY_PATH'].split(os.pathsep))
        env['LD_LIBRARY_PATH'] = os.pathsep.join(library_paths)
    if is_windows():
        cmd.insert(0, sys.executable)
    p, out = Popen_safe(cmd, cwd=cwd, env=env, stderr=subprocess.STDOUT)[0:2]
    if p.returncode != 0:
        err_msg = [f'{cmd!r} failed with status {p.returncode:d}']
        if out:
            err_msg.append(out)
        raise MesonException('\n'.join(err_msg))
    elif out:
        try:
            print(out)
        except UnicodeEncodeError:
            pass