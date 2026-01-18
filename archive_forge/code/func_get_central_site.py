from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def get_central_site(self):
    """Returns the central site of this coordination geometry."""
    return self.central_site