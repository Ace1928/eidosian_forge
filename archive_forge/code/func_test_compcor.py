import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
def test_compcor(self):
    expected_components = [[-0.1989607212, -0.5753813646], [0.5692369697, 0.5674945949], [-0.6662573243, 0.4675843432], [0.4206466244, -0.3361270124], [-0.1246655485, -0.123570561]]
    self.run_cc(CompCor(num_components=6, realigned_file=self.realigned_file, mask_files=self.mask_files, mask_index=0), expected_components)
    self.run_cc(ACompCor(num_components=6, realigned_file=self.realigned_file, mask_files=self.mask_files, mask_index=0, components_file='acc_components_file'), expected_components, 'aCompCor')