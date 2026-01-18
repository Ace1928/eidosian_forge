import os.path as op
import nibabel as nb
import numpy as np
from looseversion import LooseVersion
from ... import logging
from ..base import traits, TraitedSpec, File, isdefined
from .base import (
def resample_proxy(in_file, order=3, new_zooms=None, out_file=None):
    """
    Performs regridding of an image to set isotropic voxel sizes using dipy.
    """
    from dipy.align.reslice import reslice
    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, fext2 = op.splitext(fname)
            fext = fext2 + fext
        out_file = op.abspath('./%s_reslice%s' % (fname, fext))
    img = nb.load(in_file)
    hdr = img.header.copy()
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    im_zooms = hdr.get_zooms()[:3]
    if new_zooms is None:
        minzoom = np.array(im_zooms).min()
        new_zooms = tuple(np.ones((3,)) * minzoom)
    if np.all(im_zooms == new_zooms):
        return in_file
    data2, affine2 = reslice(data, affine, im_zooms, new_zooms, order=order)
    tmp_zooms = np.array(hdr.get_zooms())
    tmp_zooms[:3] = new_zooms[0]
    hdr.set_zooms(tuple(tmp_zooms))
    hdr.set_data_shape(data2.shape)
    hdr.set_xyzt_units('mm')
    nb.Nifti1Image(data2.astype(hdr.get_data_dtype()), affine2, hdr).to_filename(out_file)
    return (out_file, new_zooms)