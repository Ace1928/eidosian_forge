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
def iterimages(self):
    """Yield initial and displaced structures."""
    for name, atoms in self.iterdisplace():
        yield atoms