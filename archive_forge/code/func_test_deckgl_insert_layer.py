import numpy as np
import pytest
from bokeh.core.serialization import Serializer
from panel.models.deckgl import DeckGLPlot
from panel.pane import DeckGL, PaneBase, panel
def test_deckgl_insert_layer(document, comm):
    layer = {'data': [{'a': 1, 'b': 2}, {'a': 3, 'b': 7}]}
    pane = DeckGL({'layers': [layer]})
    model = pane.get_root(document, comm)
    pane.object['layers'].insert(0, {'data': [{'c': 1, 'b': 3}, {'c': 3, 'b': 9}]})
    pane.param.trigger('object')
    assert len(model.layers) == 2
    assert len(model.data_sources) == 2
    cds1, cds2 = model.data_sources
    old_data = cds1.data
    a_vals, b_vals = (old_data['a'], old_data['b'])
    layer1, layer2 = model.layers
    assert layer1['data'] == 1
    assert layer2['data'] == 0
    assert cds1.data is old_data
    assert cds1.data['a'] is a_vals
    assert cds1.data['b'] is b_vals
    assert np.array_equal(cds2.data['b'], np.array([3, 9]))
    assert np.array_equal(cds2.data['c'], np.array([1, 3]))