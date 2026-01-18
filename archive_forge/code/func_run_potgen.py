from __future__ import annotations
import os
import argparse
import subprocess
import typing as T
def run_potgen(src_sub: str, xgettext: str, pkgname: str, datadirs: str, args: T.List[str], source_root: str) -> int:
    listfile = os.path.join(src_sub, 'POTFILES.in')
    if not os.path.exists(listfile):
        listfile = os.path.join(src_sub, 'POTFILES')
        if not os.path.exists(listfile):
            print('Could not find file POTFILES in %s' % src_sub)
            return 1
    child_env = os.environ.copy()
    if datadirs:
        child_env['GETTEXTDATADIRS'] = datadirs
    ofile = os.path.join(src_sub, pkgname + '.pot')
    return subprocess.call([xgettext, '--package-name=' + pkgname, '-p', src_sub, '-f', listfile, '-D', source_root, '-k_', '-o', ofile] + args, env=child_env)