import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
def test_gh22819_cli(capfd, gh22819_cli, monkeypatch):
    """Check that module names are handled correctly
    gh-22819
    Essentially, the -m name cannot be used to import the module, so the module
    named in the .pyf needs to be used instead

    CLI :: -m and a .pyf file
    """
    ipath = Path(gh22819_cli)
    monkeypatch.setattr(sys, 'argv', f'f2py -m blah {ipath}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        gen_paths = [item.name for item in ipath.parent.rglob('*') if item.is_file()]
        assert 'blahmodule.c' not in gen_paths
        assert 'blah-f2pywrappers.f' not in gen_paths
        assert 'test_22819-f2pywrappers.f' in gen_paths
        assert 'test_22819module.c' in gen_paths
        assert 'Ignoring blah'