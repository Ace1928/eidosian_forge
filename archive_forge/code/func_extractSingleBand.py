import numpy as np
import scipy.misc as sc
import scipy.signal
import scipy.io
def extractSingleBand(self, inputimage, filtfile='sp1Filters', edges='symm', band=0, level=1):
    inputimage = inputimage.astype(np.float32)
    if filtfile == 'sp5Filters':
        lo0filt, hi0filt, lofilt, bfilts, mtx, harmonics = load_sp5filters()
    else:
        raise (NotImplementedError, 'That filter configuration is not implemnted')
    h, w = inputimage.shape
    if level == 0:
        hi0 = self.corr(inputimage, hi0filt)
        singleband = hi0
    else:
        lo0 = self.corr(inputimage, lo0filt)
        for i in range(1, level):
            lo0 = self.corr(lo0, lofilt)[::2, ::2]
        filt = bfilts[band]
        singleband = self.corr(lo0, filt)
    return singleband