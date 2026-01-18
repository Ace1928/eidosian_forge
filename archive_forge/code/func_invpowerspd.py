import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def invpowerspd(self, n):
    """autocovariance from spectral density

        scaling is correct, but n needs to be large for numerical accuracy
        maybe padding with zero in fft would be faster
        without slicing it returns 2-sided autocovariance with fftshift

        >>> ArmaFft([1, -0.5], [1., 0.4], 40).invpowerspd(2**8)[:10]
        array([ 2.08    ,  1.44    ,  0.72    ,  0.36    ,  0.18    ,  0.09    ,
                0.045   ,  0.0225  ,  0.01125 ,  0.005625])
        >>> ArmaFft([1, -0.5], [1., 0.4], 40).acovf(10)
        array([ 2.08    ,  1.44    ,  0.72    ,  0.36    ,  0.18    ,  0.09    ,
                0.045   ,  0.0225  ,  0.01125 ,  0.005625])
        """
    hw = self.fftarma(n)
    return np.real_if_close(fft.ifft(hw * hw.conj()), tol=200)[:n]