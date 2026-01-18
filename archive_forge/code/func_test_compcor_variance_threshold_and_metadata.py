import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
def test_compcor_variance_threshold_and_metadata(self):
    expected_components = [[-0.2027150345, -0.4954813834], [0.2565929051, 0.7866217875], [-0.3550986008, -0.0089784905], [0.7512786244, -0.3599828482], [-0.4500578942, 0.0778209345]]
    expected_metadata = {'component': 'CompCor00', 'mask': 'mask', 'singular_value': '4.0720553036', 'variance_explained': '0.5527211465', 'cumulative_variance_explained': '0.5527211465', 'retained': 'True'}
    ccinterface = CompCor(variance_threshold=0.7, realigned_file=self.realigned_file, mask_files=self.mask_files, mask_names=['mask'], mask_index=1, save_metadata=True)
    self.run_cc(ccinterface=ccinterface, expected_components=expected_components, expected_n_components=2, expected_metadata=expected_metadata)