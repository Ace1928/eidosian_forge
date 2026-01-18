import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
def test_compcor_no_regress_poly(self):
    self.run_cc(CompCor(num_components=6, realigned_file=self.realigned_file, mask_files=self.mask_files, mask_index=0, pre_filter=False), [[0.4451946442, -0.7683311482], [-0.4285129505, -0.0926034137], [0.5721540256, 0.5608764842], [-0.5367548139, 0.0059943226], [-0.0520809054, 0.2940637551]])