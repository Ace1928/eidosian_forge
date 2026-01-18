import os.path as op
import nibabel as nb
import numpy as np
from looseversion import LooseVersion
from ... import logging
from ..base import traits, TraitedSpec, File, isdefined
from .base import (
def nlmeans_proxy(in_file, settings, snr=None, smask=None, nmask=None, out_file=None):
    """
    Uses non-local means to denoise 4D datasets
    """
    from dipy.denoise.nlmeans import nlmeans
    from scipy.ndimage.morphology import binary_erosion
    from scipy import ndimage
    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, fext2 = op.splitext(fname)
            fext = fext2 + fext
        out_file = op.abspath('./%s_denoise%s' % (fname, fext))
    img = nb.load(in_file)
    hdr = img.header
    data = img.get_fdata()
    aff = img.affine
    if data.ndim < 4:
        data = data[..., np.newaxis]
    data = np.nan_to_num(data)
    if data.max() < 0.0001:
        raise RuntimeError('There is no signal in the image')
    df = 1.0
    if data.max() < 1000.0:
        df = 1000.0 / data.max()
        data *= df
    b0 = data[..., 0]
    if smask is None:
        smask = np.zeros_like(b0)
        smask[b0 > np.percentile(b0, 85.0)] = 1
    smask = binary_erosion(smask.astype(np.uint8), iterations=2).astype(np.uint8)
    if nmask is None:
        nmask = np.ones_like(b0, dtype=np.uint8)
        bmask = settings['mask']
        if bmask is None:
            bmask = np.zeros_like(b0)
            bmask[b0 > np.percentile(b0[b0 > 0], 10)] = 1
            label_im, nb_labels = ndimage.label(bmask)
            sizes = ndimage.sum(bmask, label_im, range(nb_labels + 1))
            maxidx = np.argmax(sizes)
            bmask = np.zeros_like(b0, dtype=np.uint8)
            bmask[label_im == maxidx] = 1
        nmask[bmask > 0] = 0
    else:
        nmask = np.squeeze(nmask)
        nmask[nmask > 0.0] = 1
        nmask[nmask < 1] = 0
        nmask = nmask.astype(bool)
    nmask = binary_erosion(nmask, iterations=1).astype(np.uint8)
    den = np.zeros_like(data)
    est_snr = True
    if snr is not None:
        snr = [snr] * data.shape[-1]
        est_snr = False
    else:
        snr = []
    for i in range(data.shape[-1]):
        d = data[..., i]
        if est_snr:
            s = np.mean(d[smask > 0])
            n = np.std(d[nmask > 0])
            snr.append(s / n)
        den[..., i] = nlmeans(d, snr[i], **settings)
    den = np.squeeze(den)
    den /= df
    nb.Nifti1Image(den.astype(hdr.get_data_dtype()), aff, hdr).to_filename(out_file)
    return (out_file, snr)