import os
import pytest
import numpy as np
import nibabel as nb
from nipype.utils.filemanip import ensure_list
from nipype.interfaces.fsl import Info
from nipype.interfaces.fsl.base import FSLCommand
def analyze_pair_image_files(outdir, filelist, shape):
    for f in ensure_list(filelist):
        hdr = nb.Nifti1Header()
        hdr.set_data_shape(shape)
        img = np.random.random(shape)
        analyze = nb.AnalyzeImage(img, np.eye(4), hdr)
        analyze.to_filename(os.path.join(outdir, f))