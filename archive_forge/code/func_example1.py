import numpy as np
from scipy import spatial as ssp
import matplotlib.pylab as plt
def example1():
    m, k = (500, 4)
    upper = 6
    scale = 10
    xs1a = np.linspace(1, upper, m)[:, np.newaxis]
    xs1 = xs1a * np.ones((1, 4)) + 1 / (1.0 + np.exp(np.random.randn(m, k)))
    xs1 /= np.std(xs1[::k, :], 0)
    y1true = np.sum(np.sin(xs1) + np.sqrt(xs1), 1)[:, np.newaxis]
    y1 = y1true + 0.25 * np.random.randn(m, 1)
    stride = 2
    gp1 = GaussProcess(xs1[::stride, :], y1[::stride, :], kernel=kernel_euclid, ridgecoeff=1e-10)
    yhatr1 = gp1.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1, 'bo', y1true, yhatr1, 'r.')
    plt.title('euclid kernel: true y versus noisy y and estimated y')
    plt.figure()
    plt.plot(y1, 'bo-', y1true, 'go-', yhatr1, 'r.-')
    plt.title('euclid kernel: true (green), noisy (blue) and estimated (red) ' + 'observations')
    gp2 = GaussProcess(xs1[::stride, :], y1[::stride, :], kernel=kernel_rbf, scale=scale, ridgecoeff=0.1)
    yhatr2 = gp2.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1, 'bo', y1true, yhatr2, 'r.')
    plt.title('rbf kernel: true versus noisy (blue) and estimated (red) observations')
    plt.figure()
    plt.plot(y1, 'bo-', y1true, 'go-', yhatr2, 'r.-')
    plt.title('rbf kernel: true (green), noisy (blue) and estimated (red) ' + 'observations')