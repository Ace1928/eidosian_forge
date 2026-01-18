import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
def test_mod_gen_gh25263(capfd, hello_world_f77, monkeypatch):
    """Check that pyf files are correctly generated with module structure
    CLI :: -m <name> -h pyf_file
    BUG: numpy-gh #20520
    """
    MNAME = 'hi'
    foutl = get_io_paths(hello_world_f77, mname=MNAME)
    ipath = foutl.finp
    monkeypatch.setattr(sys, 'argv', f'f2py {ipath} -m {MNAME} -h hi.pyf'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        with Path('hi.pyf').open() as hipyf:
            pyfdat = hipyf.read()
            assert 'python module hi' in pyfdat