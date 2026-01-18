import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
def set_fapl(plist, *args, **kwargs):
    called_with[0] = (args, kwargs)
    return _drivers['sec2'](plist)