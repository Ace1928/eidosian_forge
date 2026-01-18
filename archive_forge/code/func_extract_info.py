from ..utils import *
import numpy as np
import scipy.ndimage
import scipy.linalg
def extract_info(frame1, frame2):
    blk = 3
    sigma_nsq = 0.1
    sigma_nsqt = 0.1
    model = SpatialSteerablePyramid(height=6)
    y1 = model.extractSingleBand(frame1, filtfile='sp5Filters', band=0, level=4)
    y2 = model.extractSingleBand(frame2, filtfile='sp5Filters', band=0, level=4)
    ydiff = y1 - y2
    ss, q = est_params(y1, blk, sigma_nsq)
    ssdiff, qdiff = est_params(ydiff, blk, sigma_nsqt)
    spatial = np.multiply(q, np.log2(1 + ss))
    temporal = np.multiply(qdiff, np.multiply(np.log2(1 + ss), np.log2(1 + ssdiff)))
    return (spatial, temporal)