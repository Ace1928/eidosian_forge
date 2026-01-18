import os
import os.path as op
from collections import OrderedDict
from itertools import chain
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from .. import config, logging
from ..external.due import BibTeX
from ..interfaces.base import (
from ..utils.misc import normalize_mc_params
def compute_dvars(in_file, in_mask, remove_zerovariance=False, intensity_normalization=1000, variance_tol=0.0):
    """
    Compute the :abbr:`DVARS (D referring to temporal
    derivative of timecourses, VARS referring to RMS variance over voxels)`
    [Power2012]_.

    Particularly, the *standardized* :abbr:`DVARS (D referring to temporal
    derivative of timecourses, VARS referring to RMS variance over voxels)`
    [Nichols2013]_ are computed.

    .. [Nichols2013] Nichols T, `Notes on creating a standardized version of
         DVARS <http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/scripts/fsl/standardizeddvars.pdf>`_, 2013.

    .. note:: Implementation details

      Uses the implementation of the `Yule-Walker equations
      from nitime
      <http://nipy.org/nitime/api/generated/nitime.algorithms.autoregressive.html#nitime.algorithms.autoregressive.AR_est_YW>`_
      for the :abbr:`AR (auto-regressive)` filtering of the fMRI signal.

    :param numpy.ndarray func: functional data, after head-motion-correction.
    :param numpy.ndarray mask: a 3D mask of the brain
    :param bool output_all: write out all dvars
    :param str out_file: a path to which the standardized dvars should be saved.
    :return: the standardized DVARS

    """
    import numpy as np
    import nibabel as nb
    import warnings
    func = np.float32(nb.load(in_file).dataobj)
    mask = np.bool_(nb.load(in_mask).dataobj)
    if len(func.shape) != 4:
        raise RuntimeError('Input fMRI dataset should be 4-dimensional')
    mfunc = func[mask]
    if intensity_normalization != 0:
        mfunc = mfunc / np.median(mfunc) * intensity_normalization
    try:
        func_sd = (np.percentile(mfunc, 75, axis=1, method='lower') - np.percentile(mfunc, 25, axis=1, method='lower')) / 1.349
    except TypeError:
        func_sd = (np.percentile(mfunc, 75, axis=1, interpolation='lower') - np.percentile(mfunc, 25, axis=1, interpolation='lower')) / 1.349
    if remove_zerovariance:
        zero_variance_voxels = func_sd > variance_tol
        mfunc = mfunc[zero_variance_voxels, :]
        func_sd = func_sd[zero_variance_voxels]
    ar1 = np.apply_along_axis(_AR_est_YW, 1, regress_poly(0, mfunc, remove_mean=True)[0].astype(np.float32), 1)
    diff_sdhat = np.squeeze(np.sqrt(((1 - ar1) * 2).tolist())) * func_sd
    diff_sd_mean = diff_sdhat.mean()
    func_diff = np.diff(mfunc, axis=1)
    dvars_nstd = np.sqrt(np.square(func_diff).mean(axis=0))
    dvars_stdz = dvars_nstd / diff_sd_mean
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        diff_vx_stdz = np.square(func_diff / np.array([diff_sdhat] * func_diff.shape[-1]).T)
        dvars_vx_stdz = np.sqrt(diff_vx_stdz.mean(axis=0))
    return (dvars_stdz, dvars_nstd, dvars_vx_stdz)