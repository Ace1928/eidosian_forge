from collections import Counter
import numpy as np
from scipy import sparse
from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5
def check_weights(self):
    """Check the characteristics of the weights matrix.

        Returns
        -------
        A dict of bools containing informations about the matrix

        has_inf_val : bool
            True if the matrix has infinite values else false
        has_nan_value : bool
            True if the matrix has a "not a number" value else false
        is_not_square : bool
            True if the matrix is not square else false
        diag_is_not_zero : bool
            True if the matrix diagonal has not only zeros else false

        Examples
        --------
        >>> W = np.arange(4).reshape(2, 2)
        >>> G = graphs.Graph(W)
        >>> cw = G.check_weights()
        >>> cw == {'has_inf_val': False, 'has_nan_value': False,
        ...        'is_not_square': False, 'diag_is_not_zero': True}
        True

        """
    has_inf_val = False
    diag_is_not_zero = False
    is_not_square = False
    has_nan_value = False
    if np.isinf(self.W.sum()):
        self.logger.warning('There is an infinite value in the weight matrix!')
        has_inf_val = True
    if abs(self.W.diagonal()).sum() != 0:
        self.logger.warning('The main diagonal of the weight matrix is not 0!')
        diag_is_not_zero = True
    if self.W.get_shape()[0] != self.W.get_shape()[1]:
        self.logger.warning('The weight matrix is not square!')
        is_not_square = True
    if np.isnan(self.W.sum()):
        self.logger.warning('There is a NaN value in the weight matrix!')
        has_nan_value = True
    return {'has_inf_val': has_inf_val, 'has_nan_value': has_nan_value, 'is_not_square': is_not_square, 'diag_is_not_zero': diag_is_not_zero}