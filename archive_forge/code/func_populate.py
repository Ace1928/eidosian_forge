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
def populate(self, g):
    for i in range(100):
        if i % 10 == 0:
            g.create_group(str(i))
        else:
            g[str(i)] = [i]