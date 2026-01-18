import os
import sys
import shutil
import time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import jsonio
from ase.io.ulm import open as ulmopen
from ase.parallel import paropen, world, barrier
from ase.calculators.singlepoint import (SinglePointCalculator,
def read_info(self, framedir, name, split=None):
    """Read information about file contents without reading the data.

        Information is a dictionary containing as aminimum the shape and
        type.
        """
    fn = os.path.join(framedir, name + '.ulm')
    if split is None or os.path.exists(fn):
        with ulmopen(fn, 'r') as fd:
            info = dict()
            info['shape'] = fd.shape
            info['type'] = fd.dtype
            info['stored_as'] = fd.stored_as
            info['identical'] = fd.all_identical
        return info
    else:
        info = dict()
        for i in range(split):
            fn = os.path.join(framedir, name + '_' + str(i) + '.ulm')
            with ulmopen(fn, 'r') as fd:
                if i == 0:
                    info['shape'] = list(fd.shape)
                    info['type'] = fd.dtype
                    info['stored_as'] = fd.stored_as
                    info['identical'] = fd.all_identical
                else:
                    info['shape'][0] += fd.shape[0]
                    assert info['type'] == fd.dtype
                    info['identical'] = info['identical'] and fd.all_identical
        info['shape'] = tuple(info['shape'])
        return info