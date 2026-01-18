import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
def test_build_dir(capfd, hello_world_f90, monkeypatch):
    """Ensures that the build directory can be specified

    CLI :: --build-dir
    """
    ipath = Path(hello_world_f90)
    mname = 'blah'
    odir = 'tttmp'
    monkeypatch.setattr(sys, 'argv', f'f2py -m {mname} {ipath} --build-dir {odir}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert f'Wrote C/API module "{mname}"' in out