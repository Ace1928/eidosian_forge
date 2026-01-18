import numpy as np
from . import correlation_structures as cs
def get_y_true(self):
    if self.beta is None:
        self.y_true = self.exog.sum(1)
    else:
        self.y_true = np.dot(self.exog, self.beta)