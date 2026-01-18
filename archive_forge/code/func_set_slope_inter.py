from __future__ import annotations
import numpy as np
from .arrayproxy import ArrayProxy
from .arraywriters import ArrayWriter, WriterError, get_slope_inter, make_array_writer
from .batteryrunners import Report
from .fileholders import copy_file_map
from .spatialimages import HeaderDataError, HeaderTypeError, SpatialHeader, SpatialImage
from .volumeutils import (
from .wrapstruct import LabeledWrapStruct
def set_slope_inter(self, slope, inter=None):
    """Set slope and / or intercept into header

        Set slope and intercept for image data, such that, if the image
        data is ``arr``, then the scaled image data will be ``(arr *
        slope) + inter``

        In this case, for Analyze images, we can't store the slope or the
        intercept, so this method only checks that `slope` is None or NaN or
        1.0, and that `inter` is None or NaN or 0.

        Parameters
        ----------
        slope : None or float
            If float, value must be NaN or 1.0 or we raise a ``HeaderTypeError``
        inter : None or float, optional
            If float, value must be 0.0 or we raise a ``HeaderTypeError``
        """
    if (slope in (None, 1) or np.isnan(slope)) and (inter in (None, 0) or np.isnan(inter)):
        return
    raise HeaderTypeError('Cannot set slope != 1 or intercept != 0 for Analyze headers')