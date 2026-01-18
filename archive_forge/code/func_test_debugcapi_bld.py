import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
@pytest.mark.xfail(reason='Consistently fails on CI.')
def test_debugcapi_bld(hello_world_f90, monkeypatch):
    """Ensures that debugging wrappers work

    CLI :: --debug-capi -c
    """
    ipath = Path(hello_world_f90)
    mname = 'blah'
    monkeypatch.setattr(sys, 'argv', f'f2py -m {mname} {ipath} -c --debug-capi'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        cmd_run = shlex.split('python3 -c "import blah; blah.hi()"')
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        eout = ' Hello World\n'
        eerr = textwrap.dedent("debug-capi:Python C/API function blah.hi()\ndebug-capi:float hi=:output,hidden,scalar\ndebug-capi:hi=0\ndebug-capi:Fortran subroutine `f2pywraphi(&hi)'\ndebug-capi:hi=0\ndebug-capi:Building return value.\ndebug-capi:Python C/API function blah.hi: successful.\ndebug-capi:Freeing memory.\n        ")
        assert rout.stdout == eout
        assert rout.stderr == eerr