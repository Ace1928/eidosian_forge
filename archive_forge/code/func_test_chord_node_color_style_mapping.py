import numpy as np
import pytest
from matplotlib.collections import LineCollection, PolyCollection
from packaging.version import Version
from holoviews.core.data import Dataset
from holoviews.core.options import AbbreviatedException, Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Chord, Graph, Nodes, TriMesh, circular_layout
from holoviews.util.transform import dim
from .test_plot import TestMPLPlot, mpl_renderer
def test_chord_node_color_style_mapping(self):
    g = self.chord.opts(node_color='Label', cmap=['#FFFFFF', '#CCCCCC', '#000000'])
    plot = mpl_renderer.get_plot(g)
    arcs = plot.handles['arcs']
    nodes = plot.handles['nodes']
    self.assertEqual(np.asarray(nodes.get_array()), np.array([0, 1, 2]))
    self.assertEqual(np.asarray(arcs.get_array()), np.array([0, 1, 2]))
    self.assertEqual(nodes.get_clim(), (0, 2))
    self.assertEqual(arcs.get_clim(), (0, 2))