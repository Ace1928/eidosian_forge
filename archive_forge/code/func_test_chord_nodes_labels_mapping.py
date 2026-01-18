import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_chord_nodes_labels_mapping(self):
    g = self.chord.opts(labels='Label')
    plot = bokeh_renderer.get_plot(g)
    source = plot.handles['text_1_source']
    self.assertEqual(source.data['text'], ['A', 'B', 'C'])