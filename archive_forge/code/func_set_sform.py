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
def set_sform(self, affine, code=None, **kwargs):
    """Set sform transform from 4x4 affine

        Parameters
        ----------
        affine : None or 4x4 array
            affine transform to write into sform.  If None, only set `code`
        code : None, string or integer
            String or integer giving meaning of transform in *affine*.
            The default is None.  If code is None, then:

            * If affine is None, `code`-> 0
            * If affine not None and existing sform code in header == 0,
              `code`-> 2 (aligned)
            * If affine not None and existing sform code in header != 0,
              `code`-> existing sform code in header

        update_affine : bool, optional
            Whether to update the image affine from the header best affine
            after setting the qform.  Must be keyword argument (because of
            different position in `set_qform`). Default is True

        See also
        --------
        get_sform
        set_qform

        Examples
        --------
        >>> data = np.arange(24, dtype='f4').reshape((2,3,4))
        >>> aff = np.diag([2, 3, 4, 1])
        >>> img = Nifti1Pair(data, aff)
        >>> img.get_sform()
        array([[2., 0., 0., 0.],
               [0., 3., 0., 0.],
               [0., 0., 4., 0.],
               [0., 0., 0., 1.]])
        >>> saff, code = img.get_sform(coded=True)
        >>> saff
        array([[2., 0., 0., 0.],
               [0., 3., 0., 0.],
               [0., 0., 4., 0.],
               [0., 0., 0., 1.]])
        >>> int(code)
        2
        >>> aff2 = np.diag([3, 4, 5, 1])
        >>> img.set_sform(aff2, 'talairach')
        >>> saff, code = img.get_sform(coded=True)
        >>> np.all(saff == aff2)
        True
        >>> int(code)
        3
        """
    update_affine = kwargs.pop('update_affine', True)
    if kwargs:
        raise TypeError(f'Unexpected keyword argument(s) {kwargs}')
    self._header.set_sform(affine, code)
    if update_affine:
        if self._affine is None:
            self._affine = self._header.get_best_affine()
        else:
            self._affine[:] = self._header.get_best_affine()