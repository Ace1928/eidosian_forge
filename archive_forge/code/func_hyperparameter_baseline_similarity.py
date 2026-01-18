import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
@property
def hyperparameter_baseline_similarity(self):
    return Hyperparameter('baseline_similarity', 'numeric', self.baseline_similarity_bounds)