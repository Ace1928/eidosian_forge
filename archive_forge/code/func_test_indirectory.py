import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def test_indirectory(tmpdir):
    tmpdir.chdir()
    os.makedirs('subdir1/subdir2')
    sd1 = os.path.abspath('subdir1')
    sd2 = os.path.abspath('subdir1/subdir2')
    assert os.getcwd() == tmpdir.strpath
    with indirectory('/'):
        assert os.getcwd() == '/'
    assert os.getcwd() == tmpdir.strpath
    with indirectory('subdir1'):
        assert os.getcwd() == sd1
        with indirectory('subdir2'):
            assert os.getcwd() == sd2
            with indirectory('..'):
                assert os.getcwd() == sd1
                with indirectory('/'):
                    assert os.getcwd() == '/'
                assert os.getcwd() == sd1
            assert os.getcwd() == sd2
        assert os.getcwd() == sd1
    assert os.getcwd() == tmpdir.strpath
    try:
        with indirectory('subdir1'):
            raise ValueError('Erroring out of context')
    except ValueError:
        pass
    assert os.getcwd() == tmpdir.strpath