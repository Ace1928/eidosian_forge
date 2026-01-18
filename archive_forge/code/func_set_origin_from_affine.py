import warnings
from io import BytesIO
import numpy as np
from . import analyze  # module import
from .batteryrunners import Report
from .optpkg import optional_package
from .spatialimages import HeaderDataError, HeaderTypeError
def set_origin_from_affine(self, affine):
    """Set SPM origin to header from affine matrix.

        The ``origin`` field was read but not written by SPM99 and 2.  It was
        used for storing a central voxel coordinate, that could be used in
        aligning the image to some standard position - a proxy for a full
        translation vector that was usually stored in a separate matlab .mat
        file.

        Nifti uses the space occupied by the SPM ``origin`` field for important
        other information (the transform codes), so writing the origin will
        make the header a confusing Nifti file.  If you work with both Analyze
        and Nifti, you should probably avoid doing this.

        Parameters
        ----------
        affine : array-like, shape (4,4)
           Affine matrix to set

        Returns
        -------
        None

        Examples
        --------
        >>> hdr = Spm99AnalyzeHeader()
        >>> hdr.set_data_shape((3, 5, 7))
        >>> hdr.set_zooms((3,2,1))
        >>> hdr.get_origin_affine()
        array([[-3.,  0.,  0.,  3.],
               [ 0.,  2.,  0., -4.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])
        >>> affine = np.diag([3,2,1,1])
        >>> affine[:3,3] = [-6, -6, -4]
        >>> hdr.set_origin_from_affine(affine)
        >>> np.all(hdr['origin'][:3] == [3,4,5])
        True
        >>> hdr.get_origin_affine()
        array([[-3.,  0.,  0.,  6.],
               [ 0.,  2.,  0., -6.],
               [ 0.,  0.,  1., -4.],
               [ 0.,  0.,  0.,  1.]])
        """
    if affine.shape != (4, 4):
        raise ValueError('Need 4x4 affine to set')
    hdr = self._structarr
    RZS = affine[:3, :3]
    Z = np.sqrt(np.sum(RZS * RZS, axis=0))
    T = affine[:3, 3]
    hdr['origin'][:3] = -T / Z + 1