import pytest
import platform
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.patches as mpatches
def draw_arrow(ax, t, r):
    ax.annotate('', xy=(0.5, 0.5 + r), xytext=(0.5, 0.5), size=30, arrowprops=dict(arrowstyle=t, fc='b', ec='k'))