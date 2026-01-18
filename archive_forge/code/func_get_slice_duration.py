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
def get_slice_duration(self):
    """Get slice duration

        Returns
        -------
        slice_duration : float
            time to acquire one slice

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_dim_info(slice=2)
        >>> hdr.set_slice_duration(0.3)
        >>> print("%0.1f" % hdr.get_slice_duration())
        0.3

        Notes
        -----
        The NIfTI1 spec appears to require the slice dimension to be
        defined for slice_duration to have meaning.
        """
    _, _, slice_dim = self.get_dim_info()
    if slice_dim is None:
        raise HeaderDataError('Slice dimension must be set for duration to be valid')
    return float(self._structarr['slice_duration'])