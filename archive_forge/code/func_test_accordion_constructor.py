import pytest
from bokeh.models import Column as BkColumn, Div
import panel as pn
from panel.layout import Accordion
from panel.models import Card
def test_accordion_constructor(document, comm):
    div1 = Div()
    div2 = Div()
    accordion = Accordion(('Div1', div1), ('Div2', div2))
    p1, p2 = accordion.objects
    model = accordion.get_root(document, comm=comm)
    assert isinstance(model, BkColumn)
    assert len(model.children) == 2
    assert all((isinstance(c, Card) for c in model.children))
    card1, card2 = model.children
    assert card1.children[0].children[0].text == '&lt;h3&gt;Div1&lt;/h3&gt;'
    assert card1.children[1] is div1
    assert card2.children[0].children[0].text == '&lt;h3&gt;Div2&lt;/h3&gt;'
    assert card2.children[1] is div2