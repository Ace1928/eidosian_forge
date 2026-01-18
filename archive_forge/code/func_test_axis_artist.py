import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from mpl_toolkits.axisartist import AxisArtistHelperRectlinear
from mpl_toolkits.axisartist.axis_artist import (AxisArtist, AxisLabel,
@image_comparison(['axis_artist.png'], style='default')
def test_axis_artist():
    plt.rcParams['text.kerning_factor'] = 6
    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for loc in ('left', 'right', 'bottom'):
        helper = AxisArtistHelperRectlinear.Fixed(ax, loc=loc)
        axisline = AxisArtist(ax, helper, offset=None, axis_direction=loc)
        ax.add_artist(axisline)
    axisline.set_label('TTT')
    axisline.major_ticks.set_tick_out(False)
    axisline.label.set_pad(5)
    ax.set_ylabel('Test')