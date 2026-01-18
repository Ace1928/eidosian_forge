import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
@pytest.mark.skipif(platform.system() != 'Linux' or sys.version_info <= (3, 12), reason='Compiler and 3.12 required')
def test_no_py312_distutils_fcompiler(capfd, hello_world_f90, monkeypatch):
    """Check that no distutils imports are performed on 3.12
    CLI :: --fcompiler --help-link --backend distutils
    """
    MNAME = 'hi'
    foutl = get_io_paths(hello_world_f90, mname=MNAME)
    ipath = foutl.f90inp
    monkeypatch.setattr(sys, 'argv', f'f2py {ipath} -c --fcompiler=gfortran -m {MNAME}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert '--fcompiler cannot be used with meson' in out
    monkeypatch.setattr(sys, 'argv', f'f2py --help-link'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert 'Use --dep for meson builds' in out
    MNAME = 'hi2'
    monkeypatch.setattr(sys, 'argv', f'f2py {ipath} -c -m {MNAME} --backend distutils'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert 'Cannot use distutils backend with Python>=3.12' in out