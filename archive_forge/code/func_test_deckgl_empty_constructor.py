import numpy as np
import pytest
from bokeh.core.serialization import Serializer
from panel.models.deckgl import DeckGLPlot
from panel.pane import DeckGL, PaneBase, panel
def test_deckgl_empty_constructor(document, comm):
    pane = DeckGL()
    model = pane.get_root(document, comm)
    assert model.layers == []
    assert model.initialViewState == {}
    assert model.data == {}
    assert model.data_sources == []
    pane._cleanup(model)
    assert pane._models == {}