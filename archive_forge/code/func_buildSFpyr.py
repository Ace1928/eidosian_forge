import numpy as np
import scipy.misc as sc
import scipy.signal
import scipy.io
def buildSFpyr(self, im):
    M, N = im.shape[:2]
    log_rad, angle = self.base(M, N)
    Xrcos, Yrcos = self.rcosFn(1, -0.5)
    Yrcos = np.sqrt(Yrcos)
    YIrcos = np.sqrt(1 - Yrcos * Yrcos)
    lo0mask = self.pointOp(log_rad, YIrcos, Xrcos)
    hi0mask = self.pointOp(log_rad, Yrcos, Xrcos)
    imdft = np.fft.fftshift(np.fft.fft2(im))
    lo0dft = imdft * lo0mask
    coeff = self.buildSFpyrlevs(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height - 1)
    hi0dft = imdft * hi0mask
    hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))
    coeff.insert(0, hi0.real)
    return coeff