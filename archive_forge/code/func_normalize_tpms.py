import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
def normalize_tpms(in_files, in_mask=None, out_files=None):
    """
    Returns the input tissue probability maps (tpms, aka volume fractions)
    normalized to sum up 1.0 at each voxel within the mask.
    """
    import nibabel as nb
    import numpy as np
    import os.path as op
    in_files = np.atleast_1d(in_files).tolist()
    if out_files is None:
        out_files = []
    if len(out_files) != len(in_files):
        for i, finname in enumerate(in_files):
            fname, fext = op.splitext(op.basename(finname))
            if fext == '.gz':
                fname, fext2 = op.splitext(fname)
                fext = fext2 + fext
            out_file = op.abspath('%s_norm_%02d%s' % (fname, i, fext))
            out_files += [out_file]
    imgs = [nb.load(fim) for fim in in_files]
    if len(in_files) == 1:
        img_data = imgs[0].get_fdata(dtype=np.float32)
        img_data[img_data > 0.0] = 1.0
        hdr = imgs[0].header.copy()
        hdr.set_data_dtype(np.float32)
        nb.save(nb.Nifti1Image(img_data, imgs[0].affine, hdr), out_files[0])
        return out_files[0]
    img_data = np.stack([im.get_fdata(caching='unchanged', dtype=np.float32) for im in imgs])
    img_data[img_data < 0.0] = 0.0
    weights = np.sum(img_data, axis=0)
    msk = np.ones(imgs[0].shape)
    msk[weights <= 0] = 0
    if in_mask is not None:
        msk = np.asanyarray(nb.load(in_mask).dataobj)
        msk[msk <= 0] = 0
        msk[msk > 0] = 1
    msk = np.ma.masked_equal(msk, 0)
    for i, out_file in enumerate(out_files):
        data = np.ma.masked_equal(img_data[i], 0)
        probmap = data / weights
        hdr = imgs[i].header.copy()
        hdr.set_data_dtype('float32')
        nb.save(nb.Nifti1Image(probmap.astype(np.float32), imgs[i].affine, hdr), out_file)
    return out_files