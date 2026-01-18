import pytest
from bokeh.models import Column as BkColumn, Div
import panel as pn
from panel.layout import Accordion
from panel.models import Card
def test_accordion_cleanup_panels(document, comm, accordion):
    model = accordion.get_root(document, comm=comm)
    card1, card2 = accordion._panels.values()
    assert model.ref['id'] in card1._models
    assert model.ref['id'] in card2._models
    accordion._cleanup(model)
    assert model.ref['id'] not in card1._models
    assert model.ref['id'] not in card2._models