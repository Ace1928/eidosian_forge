from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def is_implemented(self) -> bool:
    """Returns True if this coordination geometry is implemented."""
    return bool(self.points)