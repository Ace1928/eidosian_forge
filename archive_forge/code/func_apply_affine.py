import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
def apply_affine(self, affine, lazy=True):
    """Applies an affine transformation to the streamlines.

        The transformation given by the `affine` matrix is applied after any
        other pending transformations to the streamline points.

        Parameters
        ----------
        affine : 2D array (4,4)
            Transformation matrix that will be applied on each streamline.
        lazy : True, optional
            Should always be True for :class:`LazyTractogram` object. Doing
            otherwise will raise a ValueError.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            A copy of this :class:`LazyTractogram` instance but with a
            transformation to be applied on the streamlines.
        """
    if not lazy:
        msg = 'LazyTractogram only supports lazy transformations.'
        raise ValueError(msg)
    tractogram = self.copy()
    tractogram._affine_to_apply = np.dot(affine, self._affine_to_apply)
    if tractogram.affine_to_rasmm is not None:
        tractogram.affine_to_rasmm = np.dot(self.affine_to_rasmm, np.linalg.inv(affine))
    return tractogram