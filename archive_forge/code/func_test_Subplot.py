import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from matplotlib.transforms import IdentityTransform
from mpl_toolkits.axisartist.axislines import AxesZero, SubplotZero, Subplot
from mpl_toolkits.axisartist import Axes, SubplotHost
@image_comparison(['Subplot.png'], style='default')
def test_Subplot():
    plt.rcParams['text.kerning_factor'] = 6
    fig = plt.figure()
    ax = Subplot(fig, 1, 1, 1)
    fig.add_subplot(ax)
    xx = np.arange(0, 2 * np.pi, 0.01)
    ax.plot(xx, np.sin(xx))
    ax.set_ylabel('Test')
    ax.axis['top'].major_ticks.set_tick_out(True)
    ax.axis['bottom'].major_ticks.set_tick_out(True)
    ax.axis['bottom'].set_label('Tk0')