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
def check_slicing(self, slicer: object, return_spatial: bool=False) -> tuple[slice | int | None, ...]:
    """Canonicalize slicers and check for scalar indices in spatial dims

        Parameters
        ----------
        slicer : object
            something that can be used to slice an array as in
            ``arr[sliceobj]``
        return_spatial : bool
            return only slices along spatial dimensions (x, y, z)

        Returns
        -------
        slicer : object
            Validated slicer object that will slice image's `dataobj`
            without collapsing spatial dimensions
        """
    canonical = canonical_slicers(slicer, self.img.shape)
    spatial_slices = canonical[:3]
    for subslicer in spatial_slices:
        if subslicer is None:
            raise IndexError('New axis not permitted in spatial dimensions')
        elif isinstance(subslicer, int):
            raise IndexError('Scalar indices disallowed in spatial dimensions; Use `[x]` or `x:x+1`.')
    return spatial_slices if return_spatial else canonical