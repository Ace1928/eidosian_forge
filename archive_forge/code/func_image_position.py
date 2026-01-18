import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
@one_time
def image_position(self):
    """Return position of first voxel in data block

        Adjusts Siemens mosaic position vector for bug in mosaic format
        position.  See ``dicom_mosaic`` in doc/theory for details.

        Parameters
        ----------
        None

        Returns
        -------
        img_pos : (3,) array
           position in mm of voxel (0,0,0) in Mosaic array
        """
    ipp = super().image_position
    md_rows, md_cols = (self.get('Rows'), self.get('Columns'))
    iop = self.image_orient_patient
    pix_spacing = self.get('PixelSpacing')
    if any((x is None for x in (ipp, md_rows, md_cols, iop, pix_spacing))):
        return None
    pix_spacing = np.array(list(map(float, pix_spacing)))
    md_rc = np.array([md_rows, md_cols])
    rd_rc = md_rc / self.mosaic_size
    vox_trans_fixes = (md_rc - rd_rc) / 2
    Q = np.fliplr(iop) * pix_spacing
    return ipp + np.dot(Q, vox_trans_fixes[:, None]).ravel()