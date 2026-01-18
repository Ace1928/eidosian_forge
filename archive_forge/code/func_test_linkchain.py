import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def test_linkchain(_temp_analyze_files):
    if os.name != 'posix':
        return
    orig_img, orig_hdr = _temp_analyze_files
    pth, fname = os.path.split(orig_img)
    new_img1 = os.path.join(pth, 'newfile1.img')
    new_hdr1 = os.path.join(pth, 'newfile1.hdr')
    new_img2 = os.path.join(pth, 'newfile2.img')
    new_hdr2 = os.path.join(pth, 'newfile2.hdr')
    new_img3 = os.path.join(pth, 'newfile3.img')
    new_hdr3 = os.path.join(pth, 'newfile3.hdr')
    copyfile(orig_img, new_img1)
    assert os.path.islink(new_img1)
    assert os.path.islink(new_hdr1)
    copyfile(new_img1, new_img2, copy=True)
    assert not os.path.islink(new_img2)
    assert not os.path.islink(new_hdr2)
    assert not os.path.samefile(orig_img, new_img2)
    assert not os.path.samefile(orig_hdr, new_hdr2)
    copyfile(new_img1, new_img3, copy=True, use_hardlink=True)
    assert not os.path.islink(new_img3)
    assert not os.path.islink(new_hdr3)
    assert os.path.samefile(orig_img, new_img3)
    assert os.path.samefile(orig_hdr, new_hdr3)