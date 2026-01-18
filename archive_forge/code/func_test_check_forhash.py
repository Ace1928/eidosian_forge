import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def test_check_forhash():
    fname = 'foobar'
    orig_hash = '_0x4323dbcefdc51906decd8edcb3327943'
    hashed_name = ''.join((fname, orig_hash, '.nii'))
    result, hash = check_forhash(hashed_name)
    assert result
    assert hash == [orig_hash]
    result, hash = check_forhash('foobar.nii')
    assert not result
    assert hash is None