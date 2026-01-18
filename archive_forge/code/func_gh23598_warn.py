import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
@pytest.fixture(scope='session')
def gh23598_warn(tmpdir_factory):
    """F90 file for testing warnings in gh23598"""
    fdat = util.getpath('tests', 'src', 'crackfortran', 'gh23598Warn.f90').read_text()
    fn = tmpdir_factory.getbasetemp() / 'gh23598Warn.f90'
    fn.write_text(fdat, encoding='ascii')
    return fn