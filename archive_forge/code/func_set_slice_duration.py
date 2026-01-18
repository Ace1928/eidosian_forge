from __future__ import annotations
import warnings
from io import BytesIO
import numpy as np
import numpy.linalg as npl
from . import analyze  # module import
from .arrayproxy import get_obj_dtype
from .batteryrunners import Report
from .casting import have_binary128
from .deprecated import alert_future_error
from .filebasedimages import ImageFileError, SerializableImage
from .optpkg import optional_package
from .quaternions import fillpositive, mat2quat, quat2mat
from .spatialimages import HeaderDataError
from .spm99analyze import SpmAnalyzeHeader
from .volumeutils import Recoder, endian_codes, make_dt_codes
def set_slice_duration(self, duration):
    """Set slice duration

        Parameters
        ----------
        duration : scalar
            time to acquire one slice

        Examples
        --------
        See ``get_slice_duration``
        """
    _, _, slice_dim = self.get_dim_info()
    if slice_dim is None:
        raise HeaderDataError('Slice dimension must be set for duration to be valid')
    self._structarr['slice_duration'] = duration