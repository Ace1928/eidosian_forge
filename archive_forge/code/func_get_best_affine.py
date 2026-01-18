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
def get_best_affine(self):
    """Select best of available transforms"""
    hdr = self._structarr
    if hdr['sform_code'] != 0:
        return self.get_sform()
    if hdr['qform_code'] != 0:
        return self.get_qform()
    return self.get_base_affine()