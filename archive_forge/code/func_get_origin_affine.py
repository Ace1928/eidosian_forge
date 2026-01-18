import warnings
from io import BytesIO
import numpy as np
from . import analyze  # module import
from .batteryrunners import Report
from .optpkg import optional_package
from .spatialimages import HeaderDataError, HeaderTypeError
def get_origin_affine(self):
    """Get affine from header, using SPM origin field if sensible

        The default translations are got from the ``origin``
        field, if set, or from the center of the image otherwise.

        Examples
        --------
        >>> hdr = Spm99AnalyzeHeader()
        >>> hdr.set_data_shape((3, 5, 7))
        >>> hdr.set_zooms((3, 2, 1))
        >>> hdr.default_x_flip
        True
        >>> hdr.get_origin_affine() # from center of image
        array([[-3.,  0.,  0.,  3.],
               [ 0.,  2.,  0., -4.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])
        >>> hdr['origin'][:3] = [3,4,5]
        >>> hdr.get_origin_affine() # using origin
        array([[-3.,  0.,  0.,  6.],
               [ 0.,  2.,  0., -6.],
               [ 0.,  0.,  1., -4.],
               [ 0.,  0.,  0.,  1.]])
        >>> hdr['origin'] = 0 # unset origin
        >>> hdr.set_data_shape((3, 5, 7))
        >>> hdr.get_origin_affine() # from center of image
        array([[-3.,  0.,  0.,  3.],
               [ 0.,  2.,  0., -4.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])
        """
    hdr = self._structarr
    zooms = hdr['pixdim'][1:4].copy()
    if self.default_x_flip:
        zooms[0] *= -1
    origin = hdr['origin'][:3]
    dims = hdr['dim'][1:4]
    if np.any(origin) and np.all(origin > -dims) and np.all(origin < dims * 2):
        origin = origin - 1
    else:
        origin = (dims - 1) / 2.0
    aff = np.eye(4)
    aff[:3, :3] = np.diag(zooms)
    aff[:3, -1] = -origin * zooms
    return aff