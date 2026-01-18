from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
@property
def distfactor_max(self):
    """The maximum distfactor for the perfect CoordinationGeometry (usually 1.0 for symmetric polyhedrons)."""
    dists = [np.linalg.norm(pp - self.central_site) for pp in self.points]
    return np.max(dists) / np.min(dists)