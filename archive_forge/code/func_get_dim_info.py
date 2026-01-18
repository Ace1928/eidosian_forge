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
def get_dim_info(self):
    """Gets NIfTI MRI slice etc dimension information

        Returns
        -------
        freq : {None,0,1,2}
           Which data array axis is frequency encode direction
        phase : {None,0,1,2}
           Which data array axis is phase encode direction
        slice : {None,0,1,2}
           Which data array axis is slice encode direction

        where ``data array`` is the array returned by ``get_data``

        Because NIfTI1 files are natively Fortran indexed:
          0 is fastest changing in file
          1 is medium changing in file
          2 is slowest changing in file

        ``None`` means the axis appears not to be specified.

        Examples
        --------
        See set_dim_info function

        """
    hdr = self._structarr
    info = int(hdr['dim_info'])
    freq = info & 3
    phase = info >> 2 & 3
    slice = info >> 4 & 3
    return (freq - 1 if freq else None, phase - 1 if phase else None, slice - 1 if slice else None)