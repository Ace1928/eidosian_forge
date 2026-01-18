from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
from seaborn._marks.base import (
from typing import TYPE_CHECKING
def get_transformed_path(m):
    return m.get_path().transformed(m.get_transform())