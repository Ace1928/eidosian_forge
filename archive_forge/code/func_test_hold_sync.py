import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def test_hold_sync(echo):

    class AnnoyingWidget(Widget):
        value = Float().tag(sync=True)
        other = Float().tag(sync=True)

        @observe('value')
        def _propagate_value(self, change):
            print('_propagate_value', change.new)
            if change.new == 42:
                self.value = 2
                self.other = 11
    widget = AnnoyingWidget(value=1)
    assert widget.value == 1
    widget._send = mock.MagicMock()
    widget.set_state({'value': 42})
    assert widget.value == 2
    assert widget.other == 11
    msg = {'method': 'echo_update', 'state': {'value': 42.0}, 'buffer_paths': []}
    call42 = mock.call(msg, buffers=[])
    msg = {'method': 'update', 'state': {'value': 2.0}, 'buffer_paths': []}
    call2 = mock.call(msg, buffers=[])
    msg = {'method': 'update', 'state': {'other': 11.0}, 'buffer_paths': []}
    call11 = mock.call(msg, buffers=[])
    calls = [call42, call2, call11] if echo else [call2, call11]
    widget._send.assert_has_calls(calls)