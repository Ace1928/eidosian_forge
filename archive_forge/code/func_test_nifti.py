import os
import sqlite3
from io import BytesIO
from os.path import dirname
from os.path import join as pjoin
from ..testing import suppress_warnings
import unittest
import pytest
from .. import nifti1
from ..optpkg import optional_package
def test_nifti(db):
    studies = dft.get_studies(data_dir)
    data = studies[0].series[0].as_nifti()
    assert len(data) == 352 + 2 * 256 * 256 * 2
    h = nifti1.Nifti1Header(data[:348])
    assert h.get_data_shape() == (256, 256, 2)