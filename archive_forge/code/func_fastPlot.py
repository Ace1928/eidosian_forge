from time import perf_counter
import numpy as np
import pyqtgraph as pg
def fastPlot():
    start = perf_counter()
    n = 15
    pts = 100
    x = np.linspace(0, 0.8, pts)
    y = np.random.random(size=pts) * 0.8
    shape = (n, n, pts)
    xdata = np.empty(shape)
    xdata[:] = x + np.arange(shape[1]).reshape((1, -1, 1))
    ydata = np.empty(shape)
    ydata[:] = y + np.arange(shape[0]).reshape((-1, 1, 1))
    conn = np.ones(shape, dtype=bool)
    conn[..., -1] = False
    item = pg.PlotCurveItem()
    item.setData(xdata.ravel(), ydata.ravel(), connect=conn.ravel())
    plt.addItem(item)
    dt = perf_counter() - start
    print('Create plots took: %0.3fms' % (dt * 1000))