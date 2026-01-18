import numpy as np
import scipy.misc as sc
import scipy.signal
import scipy.io
def buildLevs(self, lo0, lofilt, bfilts, edges, mHeight):
    if mHeight <= 0:
        return [lo0]
    bands = []
    for i in range(bfilts.shape[0]):
        filt = bfilts[i]
        bands.append(self.corr(lo0, filt))
    lo = self.corr(lo0, lofilt)[::2, ::2]
    bands = [bands] + self.buildLevs(lo, lofilt, bfilts, edges, mHeight - 1)
    return bands