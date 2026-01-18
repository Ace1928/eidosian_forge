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
def get_qform(self, coded=False):
    """Return 4x4 affine matrix from qform parameters in header

        Parameters
        ----------
        coded : bool, optional
            If True, return {affine or None}, and qform code.  If False, just
            return affine.  {affine or None} means, return None if qform code
            == 0, and affine otherwise.

        Returns
        -------
        affine : None or (4,4) ndarray
            If `coded` is False, always return affine reconstructed from qform
            quaternion.  If `coded` is True, return None if qform code is 0,
            else return the affine.
        code : int
            Qform code. Only returned if `coded` is True.

        See also
        --------
        set_qform
        get_sform
        """
    return self._header.get_qform(coded)