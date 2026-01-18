import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
def create_endpoints_array(fib, voxelSize):
    """Create the endpoints arrays for each fiber.

    Parameters
    ----------
    fib : array-like
      the fibers data
    voxelSize : tuple
      3-tuple containing the voxel size of the ROI image

    Returns
    -------
    endpoints : ndarray of size [#fibers, 2, 3]
      containing for each fiber the index of its first and last point in the voxelSize volume
    endpointsmm : ndarray of size [#fibers, 2, 3]
      endpoints in millimeter coordinates

    """
    n = len(fib)
    endpoints = np.zeros((n, 2, 3))
    endpointsmm = np.zeros((n, 2, 3))
    for i, fi in enumerate(fib):
        f = fi[0]
        endpoints[i, 0, :] = f[0, :]
        endpoints[i, 1, :] = f[-1, :]
        endpointsmm[i, 0, :] = f[0, :]
        endpointsmm[i, 1, :] = f[-1, :]
        endpoints[i, 0, 0] = int(endpoints[i, 0, 0] / float(voxelSize[0]))
        endpoints[i, 0, 1] = int(endpoints[i, 0, 1] / float(voxelSize[1]))
        endpoints[i, 0, 2] = int(endpoints[i, 0, 2] / float(voxelSize[2]))
        endpoints[i, 1, 0] = int(endpoints[i, 1, 0] / float(voxelSize[0]))
        endpoints[i, 1, 1] = int(endpoints[i, 1, 1] / float(voxelSize[1]))
        endpoints[i, 1, 2] = int(endpoints[i, 1, 2] / float(voxelSize[2]))
    iflogger.info('Returning the endpoint matrix')
    return (endpoints, endpointsmm)