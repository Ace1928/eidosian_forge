import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
@pytest.mark.xfail
def test_f2py_skip(capfd, retreal_f77, monkeypatch):
    """Tests that functions can be skipped
    CLI :: skip:
    """
    foutl = get_io_paths(retreal_f77, mname='test')
    ipath = foutl.finp
    toskip = 't0 t4 t8 sd s8 s4'
    remaining = 'td s0'
    monkeypatch.setattr(sys, 'argv', f'f2py {ipath} -m test skip: {toskip}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, err = capfd.readouterr()
        for skey in toskip.split():
            assert f'buildmodule: Could not found the body of interfaced routine "{skey}". Skipping.' in err
        for rkey in remaining.split():
            assert f'Constructing wrapper function "{rkey}"' in out