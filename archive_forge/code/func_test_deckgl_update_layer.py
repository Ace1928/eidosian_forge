import numpy as np
import pytest
from bokeh.core.serialization import Serializer
from panel.models.deckgl import DeckGLPlot
from panel.pane import DeckGL, PaneBase, panel
def test_deckgl_update_layer(document, comm):
    layer = {'data': [{'a': 1, 'b': 2}, {'a': 3, 'b': 7}]}
    pane = DeckGL({'layers': [layer]})
    model = pane.get_root(document, comm)
    cds = model.data_sources[0]
    old_data = cds.data
    a_vals = cds.data['a']
    layer['data'] = [{'a': 1, 'b': 3}, {'a': 3, 'b': 9}]
    pane.param.trigger('object')
    assert cds.data['a'] is a_vals
    assert cds.data is old_data
    assert np.array_equal(cds.data['b'], np.array([3, 9]))