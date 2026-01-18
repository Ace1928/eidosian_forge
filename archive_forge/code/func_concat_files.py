import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
def concat_files(bvec_file, bval_file, invert_x, invert_y, invert_z):
    bvecs = np.loadtxt(bvec_file)
    bvals = np.loadtxt(bval_file)
    if np.shape(bvecs)[0] > np.shape(bvecs)[1]:
        bvecs = np.transpose(bvecs)
    if invert_x:
        bvecs[0, :] = -bvecs[0, :]
        iflogger.info('Inverting b-vectors in the x direction')
    if invert_y:
        bvecs[1, :] = -bvecs[1, :]
        iflogger.info('Inverting b-vectors in the y direction')
    if invert_z:
        bvecs[2, :] = -bvecs[2, :]
        iflogger.info('Inverting b-vectors in the z direction')
    iflogger.info(np.shape(bvecs))
    iflogger.info(np.shape(bvals))
    encoding = np.transpose(np.vstack((bvecs, bvals)))
    _, bvec, _ = split_filename(bvec_file)
    _, bval, _ = split_filename(bval_file)
    out_encoding_file = bvec + '_' + bval + '.txt'
    np.savetxt(out_encoding_file, encoding)
    return out_encoding_file