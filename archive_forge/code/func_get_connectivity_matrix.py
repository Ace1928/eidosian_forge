import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
def get_connectivity_matrix(n_rois, list_of_roi_crossed_lists):
    connectivity_matrix = np.zeros((n_rois, n_rois), dtype=np.uint)
    for rois_crossed in list_of_roi_crossed_lists:
        for idx_i, roi_i in enumerate(rois_crossed):
            for idx_j, roi_j in enumerate(rois_crossed):
                if idx_i > idx_j:
                    if not roi_i == roi_j:
                        connectivity_matrix[roi_i - 1, roi_j - 1] += 1
    connectivity_matrix = connectivity_matrix + connectivity_matrix.T
    return connectivity_matrix