from itertools import permutations
import numpy as np
import math
from ._continuous_distns import norm
import scipy.stats
from dataclasses import dataclass
def set_k(self, k):
    """Calculate lower and upper limits of L for single row"""
    self.k = k
    self.a, self.b = (k * (k + 1) * (k + 2) // 6, k * (k + 1) * (2 * k + 1) // 6)