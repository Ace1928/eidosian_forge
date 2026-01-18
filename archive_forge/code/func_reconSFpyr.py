import numpy as np
import scipy.misc as sc
import scipy.signal
import scipy.io
def reconSFpyr(self, coeff):
    if self.nbands != len(coeff[1]):
        raise Exception('Unmatched number of orientations')
    M, N = coeff[0].shape
    log_rad, angle = self.base(M, N)
    Xrcos, Yrcos = self.rcosFn(1, -0.5)
    Yrcos = np.sqrt(Yrcos)
    YIrcos = np.sqrt(np.abs(1 - Yrcos * Yrcos))
    lo0mask = self.pointOp(log_rad, YIrcos, Xrcos)
    hi0mask = self.pointOp(log_rad, Yrcos, Xrcos)
    tempdft = self.reconSFPyrLevs(coeff[1:], log_rad, Xrcos, Yrcos, angle)
    hidft = np.fft.fftshift(np.fft.fft2(coeff[0]))
    outdft = tempdft * lo0mask + hidft * hi0mask
    return np.fft.ifft2(np.fft.ifftshift(outdft)).real