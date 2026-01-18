import json, os, subprocess, sys
from compileall import compile_file
def compileall(files):
    for f in files:
        key = f[1:11].upper()
        f = f[12:]
        ddir = None
        fullpath = absf = os.environ['MESON_INSTALL_DESTDIR_' + key] + f
        f = os.environ['MESON_INSTALL_' + key] + f
        if absf != f:
            ddir = os.path.dirname(f)
        if os.path.isdir(absf):
            for root, _, files in os.walk(absf):
                if ddir is not None:
                    ddir = root.replace(absf, f, 1)
                for dirf in files:
                    if dirf.endswith('.py'):
                        fullpath = os.path.join(root, dirf)
                        compile_file(fullpath, ddir, force=True, quiet=quiet)
        else:
            compile_file(fullpath, ddir, force=True, quiet=quiet)