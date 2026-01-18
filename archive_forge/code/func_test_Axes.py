import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from matplotlib.transforms import IdentityTransform
from mpl_toolkits.axisartist.axislines import AxesZero, SubplotZero, Subplot
from mpl_toolkits.axisartist import Axes, SubplotHost
def test_Axes():
    fig = plt.figure()
    ax = Axes(fig, [0.15, 0.1, 0.65, 0.8])
    fig.add_axes(ax)
    ax.plot([1, 2, 3], [0, 1, 2])
    ax.set_xscale('log')
    fig.canvas.draw()