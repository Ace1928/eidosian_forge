import io
from itertools import chain
import numpy as np
import pytest
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.artist as martist
import matplotlib.backend_bases as mbackend_bases
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_draw_wraper_forward_input():

    class TestKlass(martist.Artist):

        def draw(self, renderer, extra):
            return extra
    art = TestKlass()
    renderer = mbackend_bases.RendererBase()
    assert 'aardvark' == art.draw(renderer, 'aardvark')
    assert 'aardvark' == art.draw(renderer, extra='aardvark')