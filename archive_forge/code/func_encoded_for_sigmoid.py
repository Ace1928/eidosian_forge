import numpy as np
from autokeras.engine import analyser
@property
def encoded_for_sigmoid(self):
    if len(self.labels) != 2:
        return False
    return sorted(self.labels) == [0, 1]