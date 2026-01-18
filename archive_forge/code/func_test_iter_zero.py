import numpy as np
import os
import os.path
import sys
from tempfile import mkdtemp
from collections.abc import MutableMapping
from .common import ut, TestCase
import h5py
from h5py import File, Group, SoftLink, HardLink, ExternalLink
from h5py import Dataset, Datatype
from h5py import h5t
from h5py._hl.compat import filename_encode
def test_iter_zero(self):
    """ Iteration works properly for the case with no group members """
    hfile = File(self.mktemp(), 'w')
    try:
        lst = [x for x in hfile]
        self.assertEqual(lst, [])
    finally:
        hfile.close()