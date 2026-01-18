import glob
from os.path import join as pjoin
import numpy as np
from .. import Nifti1Image
from .dicomwrappers import wrapper_from_data, wrapper_from_file
def mosaic_to_nii(dcm_data):
    """Get Nifti file from Siemens

    Parameters
    ----------
    dcm_data : ``dicom.DataSet``
       DICOM header / image as read by ``dicom`` package

    Returns
    -------
    img : ``Nifti1Image``
       Nifti image object
    """
    dcm_w = wrapper_from_data(dcm_data)
    if not dcm_w.is_mosaic:
        raise DicomReadError('data does not appear to be in mosaic format')
    data = dcm_w.get_data()
    aff = np.dot(DPCS_TO_TAL, dcm_w.affine)
    return Nifti1Image(data, aff)