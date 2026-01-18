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
def test_chord_node_color_linear_style_mapping_update(self):
    hmap = HoloMap({0: self.make_chord(0), 1: self.make_chord(1)}).opts(node_color='Label', framewise=True)
    plot = mpl_renderer.get_plot(hmap)
    arcs = plot.handles['arcs']
    nodes = plot.handles['nodes']
    self.assertEqual(np.asarray(nodes.get_array()), np.array([0, 1, 2]))
    self.assertEqual(np.asarray(arcs.get_array()), np.array([0, 1, 2]))
    self.assertEqual(nodes.get_clim(), (0, 2))
    self.assertEqual(arcs.get_clim(), (0, 2))
    plot.update((1,))
    self.assertEqual(np.asarray(nodes.get_array()), np.array([1, 2, 3]))
    self.assertEqual(np.asarray(arcs.get_array()), np.array([1, 2, 3]))
    self.assertEqual(nodes.get_clim(), (1, 3))
    self.assertEqual(arcs.get_clim(), (1, 3))