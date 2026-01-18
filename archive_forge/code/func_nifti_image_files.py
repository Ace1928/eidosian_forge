import os
import pytest
import numpy as np
import nibabel as nb
from nipype.utils.filemanip import ensure_list
from nipype.interfaces.fsl import Info
from nipype.interfaces.fsl.base import FSLCommand
def nifti_image_files(outdir, filelist, shape):
    for f in ensure_list(filelist):
        img = np.random.random(shape)
        nb.Nifti1Image(img, np.eye(4), None).to_filename(os.path.join(outdir, f))