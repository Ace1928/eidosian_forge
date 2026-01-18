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
def make_chord(self, i):
    edges = [(0, 1, 1 + i), (0, 2, 2 + i), (1, 2, 3 + i)]
    nodes = Dataset([(0, 0 + i), (1, 1 + i), (2, 2 + i)], 'index', 'Label')
    return Chord((edges, nodes), vdims='weight')