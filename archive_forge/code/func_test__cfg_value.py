import os
import sys
import tempfile
from os import environ as env
from os.path import join as pjoin
from tempfile import TemporaryDirectory
import pytest
from .. import data as nibd
from ..data import (
from .test_environment import DATA_KEY, USER_KEY, with_environment
def test__cfg_value():
    assert _cfg_value('/implausible_file') == ''
    try:
        fd, tmpfile = tempfile.mkstemp()
        fobj = os.fdopen(fd, 'wt')
        fobj.write('[strange section]\n')
        fobj.write('path = /some/path\n')
        fobj.flush()
        assert _cfg_value(tmpfile) == ''
        fobj.write('[DATA]\n')
        fobj.write('funnykey = /some/path\n')
        fobj.flush()
        assert _cfg_value(tmpfile) == ''
        fobj.write('path = /some/path\n')
        fobj.flush()
        assert _cfg_value(tmpfile) == '/some/path'
        fobj.close()
    finally:
        try:
            os.unlink(tmpfile)
        except:
            pass