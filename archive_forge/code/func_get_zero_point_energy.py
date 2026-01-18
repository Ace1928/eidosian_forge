from math import pi, sqrt, log
import sys
import numpy as np
from pathlib import Path
import ase.units as units
import ase.io
from ase.parallel import world, paropen
from ase.utils.filecache import get_json_cache
from .data import VibrationsData
from collections import namedtuple
def get_zero_point_energy(self, freq=None):
    if freq:
        raise NotImplementedError()
    return self.get_vibrations().get_zero_point_energy()