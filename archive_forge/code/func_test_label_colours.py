import numpy as np
import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.table import CustomCell, Table
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
@image_comparison(['table_labels.png'])
def test_label_colours():
    dim = 3
    c = np.linspace(0, 1, dim)
    colours = plt.cm.RdYlGn(c)
    cellText = [['1'] * dim] * dim
    fig = plt.figure()
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.axis('off')
    ax1.table(cellText=cellText, rowColours=colours, loc='best')
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.axis('off')
    ax2.table(cellText=cellText, rowColours=colours, rowLabels=['Header'] * dim, loc='best')
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.axis('off')
    ax3.table(cellText=cellText, colColours=colours, loc='best')
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.axis('off')
    ax4.table(cellText=cellText, colColours=colours, colLabels=['Header'] * dim, loc='best')