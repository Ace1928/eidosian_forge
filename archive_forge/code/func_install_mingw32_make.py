import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from collections import OrderedDict
from time import sleep
from typing import Any, Dict, List
from cmdstanpy import _DOT_CMDSTAN
from cmdstanpy.utils import pushd, validate_dir, wrap_url_progress_hook
def install_mingw32_make(toolchain_loc: str, verbose: bool=False) -> None:
    """Install mingw32-make for Windows RTools 4.0."""
    os.environ['PATH'] = ';'.join(list(OrderedDict.fromkeys([os.path.join(toolchain_loc, 'mingw_64' if IS_64BITS else 'mingw_32', 'bin'), os.path.join(toolchain_loc, 'usr', 'bin')] + os.environ.get('PATH', '').split(';'))))
    cmd = ['pacman', '-Sy', 'mingw-w64-x86_64-make' if IS_64BITS else 'mingw-w64-i686-make', '--noconfirm']
    with pushd('.'):
        print(' '.join(cmd))
        proc = subprocess.Popen(cmd, cwd=None, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)
        while proc.poll() is None:
            if proc.stdout:
                output = proc.stdout.readline().decode('utf-8').strip()
                if output and verbose:
                    print(output, flush=True)
        _, stderr = proc.communicate()
        if proc.returncode:
            print('mingw32-make installation failed: returncode={}'.format(proc.returncode))
            if stderr:
                print(stderr.decode('utf-8').strip())
            sys.exit(3)
    print('Installed mingw32-make.exe')