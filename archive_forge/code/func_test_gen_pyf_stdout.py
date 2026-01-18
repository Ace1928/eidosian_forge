import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
def test_gen_pyf_stdout(capfd, hello_world_f90, monkeypatch):
    """Ensures that a signature file can be dumped to stdout
    CLI :: -h
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, 'argv', f'f2py -h stdout {ipath}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert 'Saving signatures to file' in out
        assert 'function hi() ! in ' in out