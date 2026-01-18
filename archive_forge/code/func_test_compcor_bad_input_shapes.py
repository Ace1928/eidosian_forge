import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
def test_compcor_bad_input_shapes(self):
    shape_less_than = (1, 2, 2, 5)
    shape_more_than = (3, 3, 3, 5)
    for data_shape in (shape_less_than, shape_more_than):
        data_file = utils.save_toy_nii(np.zeros(data_shape), 'temp.nii')
        interface = CompCor(realigned_file=data_file, mask_files=self.mask_files[0])
        with pytest.raises(ValueError):
            interface.run()