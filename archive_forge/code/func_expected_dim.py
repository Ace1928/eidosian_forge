import numpy as np
from autokeras.engine import analyser
def expected_dim(self):
    if len(self.shape) == 1:
        return 1
    return self.shape[1]