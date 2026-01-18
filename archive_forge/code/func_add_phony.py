import weakref
from collections import OrderedDict
from collections.abc import MutableMapping
import h5py
import numpy as np
def add_phony(self, name, size):
    self._objects[name] = Dimension(self._group, name, size, create_h5ds=False, phony=True)