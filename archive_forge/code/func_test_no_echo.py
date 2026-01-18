import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def test_no_echo(echo):

    class ValueWidget(Widget):
        value = Float().tag(sync=True, echo_update=False)
    widget = ValueWidget(value=1)
    assert widget.value == 1
    widget._send = mock.MagicMock()
    widget._handle_msg({'content': {'data': {'method': 'update', 'state': {'value': 42}}}})
    assert widget.value == 42
    widget._send.assert_not_called()
    widget.value = 43
    widget._send.assert_has_calls([mock.call({'method': 'update', 'state': {'value': 43.0}, 'buffer_paths': []}, buffers=[])])