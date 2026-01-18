import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
def get_rois_crossed(pointsmm, roiData, voxelSize):
    n_points = len(pointsmm)
    rois_crossed = []
    for j in range(0, n_points):
        x = int(pointsmm[j, 0] / float(voxelSize[0]))
        y = int(pointsmm[j, 1] / float(voxelSize[1]))
        z = int(pointsmm[j, 2] / float(voxelSize[2]))
        if not roiData[x, y, z] == 0:
            rois_crossed.append(roiData[x, y, z])
    rois_crossed = list(dict.fromkeys(rois_crossed).keys())
    return rois_crossed