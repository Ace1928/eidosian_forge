from __future__ import annotations
from typing import TYPE_CHECKING
from warnings import warn
import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineWarning
from .stat import stat
def scale_simp(x: FloatArray, center: FloatArray, n: int, p: int):
    return x - np.repeat([center], n, axis=0)