import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
def create_allpoints_cmat(streamlines, roiData, voxelSize, n_rois):
    """Create the intersection arrays for each fiber"""
    n_fib = len(streamlines)
    pc = -1
    final_fiber_ids = []
    list_of_roi_crossed_lists = []
    for i, fiber in enumerate(streamlines):
        pcN = int(round(float(100 * i) / n_fib))
        if pcN > pc and pcN % 1 == 0:
            pc = pcN
            print('%4.0f%%' % pc)
        rois_crossed = get_rois_crossed(fiber[0], roiData, voxelSize)
        if len(rois_crossed) > 0:
            list_of_roi_crossed_lists.append(list(rois_crossed))
            final_fiber_ids.append(i)
    connectivity_matrix = get_connectivity_matrix(n_rois, list_of_roi_crossed_lists)
    dis = n_fib - len(final_fiber_ids)
    iflogger.info('Found %i (%f percent out of %i fibers) fibers that start or terminate in a voxel which is not labeled. (orphans)', dis, dis * 100.0 / n_fib, n_fib)
    iflogger.info('Valid fibers: %i (%f percent)', n_fib - dis, 100 - dis * 100.0 / n_fib)
    iflogger.info('Returning the intersecting point connectivity matrix')
    return (connectivity_matrix, final_fiber_ids)