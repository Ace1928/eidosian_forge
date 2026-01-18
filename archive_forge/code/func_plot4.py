import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def plot4(self, fig=None, nobs=100, nacf=20, nfreq=100):
    """Plot results"""
    rvs = self.generate_sample(nsample=100, burnin=500)
    acf = self.acf(nacf)[:nacf]
    pacf = self.pacf(nacf)
    w = np.linspace(0, np.pi, nfreq)
    spdr, wr = self.spdroots(w)
    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(rvs)
    ax.set_title('Random Sample \nar={}, ma={}'.format(self.ar, self.ma))
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(acf)
    ax.set_title('Autocorrelation \nar={}, ma={!r}s'.format(self.ar, self.ma))
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(wr, spdr)
    ax.set_title('Power Spectrum \nar={}, ma={}'.format(self.ar, self.ma))
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(pacf)
    ax.set_title('Partial Autocorrelation \nar={}, ma={}'.format(self.ar, self.ma))
    return fig