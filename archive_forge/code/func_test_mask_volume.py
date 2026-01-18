import numpy as np
from .. import Nifti1Image, imagestats
def test_mask_volume():
    mask_data = np.zeros((20, 20, 20), dtype='u1')
    mask_data[5:15, 5:15, 5:15] = 1
    img = Nifti1Image(mask_data, np.eye(4))
    vol_mm3 = imagestats.mask_volume(img)
    vol_vox = imagestats.count_nonzero_voxels(img)
    assert vol_mm3 == 1000.0
    assert vol_vox == 1000