import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def test_copyfile_true(_temp_analyze_files):
    orig_img, orig_hdr = _temp_analyze_files
    pth, fname = os.path.split(orig_img)
    new_img = os.path.join(pth, 'newfile.img')
    new_hdr = os.path.join(pth, 'newfile.hdr')
    copyfile(orig_img, new_img, copy=True)
    assert os.path.exists(new_img)
    assert os.path.exists(new_hdr)