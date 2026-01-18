from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
def test_engformatter_usetex_useMathText():
    fig, ax = plt.subplots()
    ax.plot([0, 500, 1000], [0, 500, 1000])
    ax.set_xticks([0, 500, 1000])
    for formatter in (mticker.EngFormatter(usetex=True), mticker.EngFormatter(useMathText=True)):
        ax.xaxis.set_major_formatter(formatter)
        fig.canvas.draw()
        x_tick_label_text = [labl.get_text() for labl in ax.get_xticklabels()]
        assert x_tick_label_text == ['$0$', '$500$', '$1$ k']