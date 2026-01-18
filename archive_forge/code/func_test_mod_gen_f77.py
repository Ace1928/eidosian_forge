import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
def test_mod_gen_f77(capfd, hello_world_f90, monkeypatch):
    """Checks the generation of files based on a module name
    CLI :: -m
    """
    MNAME = 'hi'
    foutl = get_io_paths(hello_world_f90, mname=MNAME)
    ipath = foutl.f90inp
    monkeypatch.setattr(sys, 'argv', f'f2py {ipath} -m {MNAME}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
    assert Path.exists(foutl.cmodf)
    assert Path.exists(foutl.wrap77)