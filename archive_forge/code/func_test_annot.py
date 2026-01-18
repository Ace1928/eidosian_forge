import getpass
import hashlib
import os
import struct
import time
import unittest
from os.path import isdir
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
from ...fileslice import strided_scalar
from ...testing import clear_and_catch_warnings
from ...tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from ...tmpdirs import InTemporaryDirectory
from .. import (
from ..io import _pack_rgb
@freesurfer_test
def test_annot():
    """Test IO of .annot against freesurfer example data."""
    annots = ['aparc', 'aparc.a2005s']
    for a in annots:
        annot_path = pjoin(data_path, 'label', f'lh.{a}.annot')
        labels, ctab, names = read_annot(annot_path)
        assert labels.shape == (163842,)
        assert ctab.shape == (len(names), 5)
        labels_orig = None
        if a == 'aparc':
            labels_orig, _, _ = read_annot(annot_path, orig_ids=True)
            np.testing.assert_array_equal(labels == -1, labels_orig == 0)
            content_hash = hashlib.md5(Path(annot_path).read_bytes()).hexdigest()
            if content_hash == 'bf0b488994657435cdddac5f107d21e8':
                assert np.sum(labels_orig == 0) == 13887
            elif content_hash == 'd4f5b7cbc2ed363ac6fcf89e19353504':
                assert np.sum(labels_orig == 1639705) == 13327
            else:
                raise RuntimeError('Unknown freesurfer file. Please report the problem to the maintainer of nibabel.')
        with InTemporaryDirectory():
            annot_path = 'test'
            write_annot(annot_path, labels, ctab, names)
            labels2, ctab2, names2 = read_annot(annot_path)
            if labels_orig is not None:
                labels_orig_2, _, _ = read_annot(annot_path, orig_ids=True)
        assert np.array_equal(labels, labels2)
        if labels_orig is not None:
            assert np.array_equal(labels_orig, labels_orig_2)
        assert np.array_equal(ctab, ctab2)
        assert names == names2