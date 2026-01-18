import numpy as np
import pytest
from bokeh.core.serialization import Serializer
from panel.models.deckgl import DeckGLPlot
from panel.pane import DeckGL, PaneBase, panel
def test_deckgl_update_layer_columns(document, comm):
    layer = {'data': [{'a': 1, 'b': 2}, {'a': 3, 'b': 7}]}
    pane = DeckGL({'layers': [layer]})
    model = pane.get_root(document, comm)
    cds = model.data_sources[0]
    old_data = cds.data
    layer['data'] = [{'c': 1, 'b': 3}, {'c': 3, 'b': 9}]
    pane.param.trigger('object')
    assert 'a' not in cds.data
    assert cds.data is not old_data
    assert np.array_equal(cds.data['b'], np.array([3, 9]))
    assert np.array_equal(cds.data['c'], np.array([1, 3]))