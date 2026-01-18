import numpy as np
from collections.abc import MutableMapping
from .common import TestCase, ut
import h5py
from h5py import File
from h5py import h5a,  h5t
from h5py import AttributeManager
def fill_attrs2(self, track_order):
    group = self.f.create_group('test', track_order=track_order)
    for i in range(12):
        group.attrs[str(i)] = i
    return group