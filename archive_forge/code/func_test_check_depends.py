import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def test_check_depends(tmpdir):

    def touch(fname):
        with open(fname, 'a'):
            os.utime(fname, None)
    dependencies = [tmpdir.join(str(i)).strpath for i in range(3)]
    targets = [tmpdir.join(str(i)).strpath for i in range(3, 6)]
    for dep in dependencies:
        touch(dep)
    time.sleep(1)
    for tgt in targets:
        touch(tgt)
    assert check_depends(targets, dependencies)
    time.sleep(1)
    touch(dependencies[0])
    assert not check_depends(targets, dependencies)
    os.unlink(dependencies[0])
    try:
        check_depends(targets, dependencies)
    except OSError:
        pass
    else:
        assert False, 'Should raise OSError on missing dependency'