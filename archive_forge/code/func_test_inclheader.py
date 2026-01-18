import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
def test_inclheader(capfd, hello_world_f90, monkeypatch):
    """Add to the include directories

    CLI :: -include
    TODO: Document this in the help string
    """
    ipath = Path(hello_world_f90)
    mname = 'blah'
    monkeypatch.setattr(sys, 'argv', f'f2py -m {mname} {ipath} -include<stdbool.h> -include<stdio.h> '.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        with Path(f'./{mname}module.c').open() as ocmod:
            ocmr = ocmod.read()
            assert '#include <stdbool.h>' in ocmr
            assert '#include <stdio.h>' in ocmr