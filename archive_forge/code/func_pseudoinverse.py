from . import utils
from scipy import sparse
from sklearn import decomposition
from sklearn import random_projection
import numpy as np
import pandas as pd
import sklearn.base
import warnings
@property
def pseudoinverse(self):
    """Pseudoinverse of the random projection.

        This inverts the projection operation for any vector in the span of the
        random projection. For small enough `eps`, this should be close to the
        correct inverse.
        """
    try:
        return self._pseudoinverse
    except AttributeError:
        if self.orthogonalize:
            self._pseudoinverse = self.components_
        else:
            self._pseudoinverse = np.linalg.pinv(self.components_.T)
        return self._pseudoinverse