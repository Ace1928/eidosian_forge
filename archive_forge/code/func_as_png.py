import contextlib
import getpass
import logging
import os
import sqlite3
import tempfile
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy
from nibabel.optpkg import optional_package
from .nifti1 import Nifti1Header
def as_png(self, index=None, scale_to_slice=True):
    import PIL.Image
    if hasattr(PIL.Image, 'frombytes'):
        frombytes = PIL.Image.frombytes
    else:
        frombytes = PIL.Image.fromstring
    if index is None:
        index = len(self.storage_instances) // 2
    d = self.storage_instances[index].dicom()
    data = d.pixel_array.copy()
    if self.bits_allocated != 16:
        raise VolumeError('unsupported bits allocated')
    if self.bits_stored != 12:
        raise VolumeError('unsupported bits stored')
    data = data / 16
    if scale_to_slice:
        min = data.min()
        max = data.max()
        data = data * 255 / (max - min)
    data = data.astype(numpy.uint8)
    im = frombytes('L', (self.rows, self.columns), data.tobytes())
    s = BytesIO()
    im.save(s, 'PNG')
    return s.getvalue()