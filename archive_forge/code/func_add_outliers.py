from dataclasses import dataclass
from functools import reduce  # Required in Python 3
import operator
from typing import Callable, Optional, Tuple
import warnings
from warnings import warn
import torch
import bitsandbytes.functional as F
def add_outliers(self, outlier_idx, feature_dim):
    if self.model_dim is None:
        self.model_dim = feature_dim
    if feature_dim != self.model_dim:
        return
    self.outliers.update(outlier_idx.tolist())