import numpy as np
from autokeras.engine import analyser
@property
def encoded_for_softmax(self):
    return len(self.shape) > 1 and self.shape[1] > 1