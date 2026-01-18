import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import importlib
from typing import Set, Optional, List
import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel
from gensim.utils import SaveLoad
def mass_masking(a, threshold=None):
    """Original masking method. Returns a new binary mask."""
    if threshold is None:
        threshold = 0.95
    sorted_a = np.sort(a)[::-1]
    largest_mass = sorted_a.cumsum() < threshold
    smallest_valid = sorted_a[largest_mass][-1]
    return a >= smallest_valid