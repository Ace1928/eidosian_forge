import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
def test_lower_cmod(capfd, hello_world_f77, monkeypatch):
    """Lowers cases by flag or when -h is present

    CLI :: --[no-]lower
    """
    foutl = get_io_paths(hello_world_f77, mname='test')
    ipath = foutl.finp
    capshi = re.compile('HI\\(\\)')
    capslo = re.compile('hi\\(\\)')
    monkeypatch.setattr(sys, 'argv', f'f2py {ipath} -m test --lower'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is not None
        assert capshi.search(out) is None
    monkeypatch.setattr(sys, 'argv', f'f2py {ipath} -m test --no-lower'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is None
        assert capshi.search(out) is not None