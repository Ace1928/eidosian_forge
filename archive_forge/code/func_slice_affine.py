from __future__ import annotations
import io
import typing as ty
from collections.abc import Sequence
from typing import Literal
import numpy as np
from .arrayproxy import ArrayLike
from .casting import sctypes_aliases
from .dataobj_images import DataobjImage
from .filebasedimages import FileBasedHeader, FileBasedImage
from .fileholders import FileMap
from .fileslice import canonical_slicers
from .orientations import apply_orientation, inv_ornt_aff
from .viewers import OrthoSlicer3D
from .volumeutils import shape_zoom_affine
def slice_affine(self, slicer: object) -> np.ndarray:
    """Retrieve affine for current image, if sliced by a given index

        Applies scaling if down-sampling is applied, and adjusts the intercept
        to account for any cropping.

        Parameters
        ----------
        slicer : object
            something that can be used to slice an array as in
            ``arr[sliceobj]``

        Returns
        -------
        affine : (4,4) ndarray
            Affine with updated scale and intercept
        """
    slicer = self.check_slicing(slicer, return_spatial=True)
    transform = np.eye(4, dtype=int)
    for i, subslicer in enumerate(slicer):
        if isinstance(subslicer, slice):
            if subslicer.step == 0:
                raise ValueError('slice step cannot be 0')
            transform[i, i] = subslicer.step if subslicer.step is not None else 1
            transform[i, 3] = subslicer.start or 0
    return self.img.affine.dot(transform)