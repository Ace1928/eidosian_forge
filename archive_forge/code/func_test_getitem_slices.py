from unittest import TestCase
from unittest import mock
import pytest
import traitlets
import ipywidgets as widgets
from ipywidgets.widgets.widget_templates import LayoutProperties
def test_getitem_slices(self):
    """test retrieving widgets with slices"""
    box = widgets.GridspecLayout(2, 3)
    button1 = widgets.Button()
    box[:2, 0] = button1
    assert box[:2, 0] is button1
    box = widgets.GridspecLayout(2, 3)
    button1 = widgets.Button()
    button2 = widgets.Button()
    box[0, 0] = button1
    box[1, 0] = button2
    assert box[0, 0] is button1
    assert box[1, 0] is button2
    with pytest.raises(TypeError, match='The slice spans'):
        button = box[:2, 0]