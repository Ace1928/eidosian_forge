import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
def test_find_route():
    graph = {'A': ('B', 'C'), 'C': ('D',), 'D': ('E', 'F', 'G'), 'F': ('H',), 'G': ('I',)}
    assert find_route(graph, 'A', 'I') == ['C', 'D', 'G', 'I']
    assert find_route(graph, 'B', 'I') is None
    assert find_route(graph, 'D', 'H') == ['F', 'H']