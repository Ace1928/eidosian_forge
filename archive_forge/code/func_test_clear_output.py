import sys
from unittest import TestCase
from contextlib import contextmanager
from IPython.display import Markdown, Image
from ipywidgets import widget_output
def test_clear_output(self):
    msg_id = 'msg-id'
    get_ipython = self._mock_get_ipython(msg_id)
    clear_output = self._mock_clear_output()
    with self._mocked_ipython(get_ipython, clear_output):
        widget = widget_output.Output()
        widget.clear_output(wait=True)
    assert len(clear_output.calls) == 1
    assert clear_output.calls[0] == ((), {'wait': True})