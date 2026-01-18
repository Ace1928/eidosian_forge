import os
import tempfile
import unittest
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.compat.py3k import asbytes
import nibabel as nib
from nibabel.testing import clear_and_catch_warnings, data_path, error_warnings
from nibabel.tmpdirs import InTemporaryDirectory
from .. import FORMATS, trk
from ..tractogram import LazyTractogram, Tractogram
from ..tractogram_file import ExtensionWarning, TractogramFile
from .test_tractogram import assert_tractogram_equal
def test_is_supported_detect_format(tmp_path):
    f = BytesIO()
    assert not nib.streamlines.is_supported(f)
    assert not nib.streamlines.is_supported('')
    assert nib.streamlines.detect_format(f) is None
    assert nib.streamlines.detect_format('') is None
    for tfile_cls in FORMATS.values():
        f = BytesIO()
        f.write(asbytes(tfile_cls.MAGIC_NUMBER))
        f.seek(0, os.SEEK_SET)
        assert nib.streamlines.is_supported(f)
        assert nib.streamlines.detect_format(f) is tfile_cls
    for tfile_cls in FORMATS.values():
        fpath = tmp_path / 'test.txt'
        with open(fpath, 'w+b') as f:
            f.write(asbytes(tfile_cls.MAGIC_NUMBER))
            f.seek(0, os.SEEK_SET)
            assert nib.streamlines.is_supported(f)
            assert nib.streamlines.detect_format(f) is tfile_cls
    for ext, tfile_cls in FORMATS.items():
        fpath = tmp_path / f'test{ext}'
        with open(fpath, 'w+b') as f:
            f.write(b'pass')
            f.seek(0, os.SEEK_SET)
            assert not nib.streamlines.is_supported(f)
            assert nib.streamlines.detect_format(f) is None
    f = 'my_tractogram.asd'
    assert not nib.streamlines.is_supported(f)
    assert nib.streamlines.detect_format(f) is None
    for ext, tfile_cls in FORMATS.items():
        f = 'my_tractogram' + ext
        assert nib.streamlines.is_supported(f)
        assert nib.streamlines.detect_format(f) == tfile_cls
    for ext, tfile_cls in FORMATS.items():
        f = 'my_tractogram' + ext.upper()
        assert nib.streamlines.detect_format(f) is tfile_cls