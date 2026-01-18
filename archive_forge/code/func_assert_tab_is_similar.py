import pytest
from bokeh.models import Column as BkColumn, Div
import panel as pn
from panel.layout import Accordion
from panel.models import Card
def assert_tab_is_similar(tab1, tab2):
    """Helper function to check tab match"""
    assert tab1.child is tab2.child
    assert tab1.name == tab2.name
    assert tab1.title == tab2.title