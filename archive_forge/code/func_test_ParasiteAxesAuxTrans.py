import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from matplotlib.transforms import IdentityTransform
from mpl_toolkits.axisartist.axislines import AxesZero, SubplotZero, Subplot
from mpl_toolkits.axisartist import Axes, SubplotHost
@image_comparison(['ParasiteAxesAuxTrans_meshplot.png'], remove_text=True, style='default', tol=0.075)
def test_ParasiteAxesAuxTrans():
    data = np.ones((6, 6))
    data[2, 2] = 2
    data[0, :] = 0
    data[-2, :] = 0
    data[:, 0] = 0
    data[:, -2] = 0
    x = np.arange(6)
    y = np.arange(6)
    xx, yy = np.meshgrid(x, y)
    funcnames = ['pcolor', 'pcolormesh', 'contourf']
    fig = plt.figure()
    for i, name in enumerate(funcnames):
        ax1 = SubplotHost(fig, 1, 3, i + 1)
        fig.add_subplot(ax1)
        ax2 = ax1.get_aux_axes(IdentityTransform(), viewlim_mode=None)
        if name.startswith('pcolor'):
            getattr(ax2, name)(xx, yy, data[:-1, :-1])
        else:
            getattr(ax2, name)(xx, yy, data)
        ax1.set_xlim((0, 5))
        ax1.set_ylim((0, 5))
    ax2.contour(xx, yy, data, colors='k')