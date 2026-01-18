import collections
from functools import reduce, singledispatch
from typing import (Any, Dict, Iterable, List, Optional,
import numpy as np
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData, Info
from ase.utils.plotting import SimplePlottingAxes
def get_all_weights(self) -> Sequence[Sequence[float]]:
    return self._weights.copy()