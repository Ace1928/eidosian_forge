import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
@pytest.fixture(scope='session')
def hello_world_f90(tmpdir_factory):
    """Generates a single f90 file for testing"""
    fdat = util.getpath('tests', 'src', 'cli', 'hiworld.f90').read_text()
    fn = tmpdir_factory.getbasetemp() / 'hello.f90'
    fn.write_text(fdat, encoding='ascii')
    return fn