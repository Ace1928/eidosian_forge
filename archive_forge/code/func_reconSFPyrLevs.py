import numpy as np
import scipy.misc as sc
import scipy.signal
import scipy.io
def reconSFPyrLevs(self, coeff, log_rad, Xrcos, Yrcos, angle):
    if len(coeff) == 1:
        return np.fft.fftshift(np.fft.fft2(coeff[0]))
    else:
        Xrcos = Xrcos - 1
        himask = self.pointOp(log_rad, Yrcos, Xrcos)
        lutsize = 1024
        Xcosn = np.pi * np.array(range(-(2 * lutsize + 1), lutsize + 2)) / lutsize
        order = self.nbands - 1
        const = np.power(2, 2 * order) * np.square(sc.factorial(order)) / (self.nbands * sc.factorial(2 * order))
        Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)
        orientdft = np.zeros(coeff[0][0].shape, 'complex')
        for b in range(int(self.nbands)):
            anglemask = self.pointOp(angle, Ycosn, Xcosn + np.pi * b / self.nbands)
            banddft = np.fft.fftshift(np.fft.fft2(coeff[0][b]))
            orientdft += np.complex(0, 1) ** order * banddft * anglemask * himask
        dims = np.array(coeff[0][0].shape)
        lostart = np.ceil((dims + 0.5) / 2) - np.ceil((np.ceil((dims - 0.5) / 2) + 0.5) / 2)
        loend = lostart + np.ceil((dims - 0.5) / 2)
        nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
        nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
        YIrcos = np.sqrt(np.abs(1 - Yrcos * Yrcos))
        lomask = self.pointOp(nlog_rad, YIrcos, Xrcos)
        nresdft = self.reconSFPyrLevs(coeff[1:], nlog_rad, Xrcos, Yrcos, nangle)
        res = np.fft.fftshift(np.fft.fft2(nresdft))
        resdft = np.zeros(dims, 'complex')
        resdft[lostart[0]:loend[0], lostart[1]:loend[1]] = nresdft * lomask
        return resdft + orientdft