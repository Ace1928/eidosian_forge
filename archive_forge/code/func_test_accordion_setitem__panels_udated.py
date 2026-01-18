import pytest
from bokeh.models import Column as BkColumn, Div
import panel as pn
from panel.layout import Accordion
from panel.models import Card
def test_accordion_setitem__panels_udated(document, comm, accordion):
    accordion.get_root(document, comm=comm)
    accordion[:] = [('Card1', '1'), ('Card2', '2'), ('Card3', '3')]
    assert len(accordion._panels) == 3