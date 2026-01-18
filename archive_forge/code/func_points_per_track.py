import os.path as op
import nibabel as nb
import numpy as np
from nibabel.volumeutils import native_code
from nibabel.orientations import aff2axcodes
from ... import logging
from ...utils.filemanip import split_filename
from ..base import TraitedSpec, File, isdefined
from ..dipy.base import DipyBaseInterface, HAVE_DIPY as have_dipy
def points_per_track(offset):
    track_points = []
    iflogger.info('Identifying the number of points per tract...')
    all_str = fileobj.read()
    num_triplets = int(len(all_str) / bytesize)
    pts = np.ndarray(shape=(num_triplets, pt_cols), dtype='f4', buffer=all_str)
    nonfinite_list = np.where(np.invert(np.isfinite(pts[:, 2])))
    nonfinite_list = list(nonfinite_list[0])[0:-1]
    for idx, value in enumerate(nonfinite_list):
        if idx == 0:
            track_points.append(nonfinite_list[idx])
        else:
            track_points.append(nonfinite_list[idx] - nonfinite_list[idx - 1] - 1)
    return (track_points, nonfinite_list)