import glob
from os.path import join as pjoin
import numpy as np
from .. import Nifti1Image
from .dicomwrappers import wrapper_from_data, wrapper_from_file
def read_mosaic_dwi_dir(dicom_path, globber='*.dcm', dicom_kwargs=None):
    return read_mosaic_dir(dicom_path, globber, check_is_dwi=True, dicom_kwargs=dicom_kwargs)