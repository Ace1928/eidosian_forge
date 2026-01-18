import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
def test_colorbar_align():
    for location in ['right', 'left', 'top', 'bottom']:
        fig, axs = plt.subplots(2, 2, layout='constrained')
        cbs = []
        for nn, ax in enumerate(axs.flat):
            ax.tick_params(direction='in')
            pc = example_pcolor(ax)
            cb = fig.colorbar(pc, ax=ax, location=location, shrink=0.6, pad=0.04)
            cbs += [cb]
            cb.ax.tick_params(direction='in')
            if nn != 1:
                cb.ax.xaxis.set_ticks([])
                cb.ax.yaxis.set_ticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.1, wspace=0.1)
        fig.draw_without_rendering()
        if location in ['left', 'right']:
            np.testing.assert_allclose(cbs[0].ax.get_position().x0, cbs[2].ax.get_position().x0)
            np.testing.assert_allclose(cbs[1].ax.get_position().x0, cbs[3].ax.get_position().x0)
        else:
            np.testing.assert_allclose(cbs[0].ax.get_position().y0, cbs[1].ax.get_position().y0)
            np.testing.assert_allclose(cbs[2].ax.get_position().y0, cbs[3].ax.get_position().y0)