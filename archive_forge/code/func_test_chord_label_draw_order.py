import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_chord_label_draw_order(self):
    g = self.chord.opts(labels='Label')
    plot = bokeh_renderer.get_plot(g)
    renderers = plot.state.renderers
    graph_renderer = plot.handles['glyph_renderer']
    label_renderer = plot.handles['text_1_glyph_renderer']
    self.assertTrue(renderers.index(graph_renderer) < renderers.index(label_renderer))