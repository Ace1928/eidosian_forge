import inspect
import pytest
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display
from IPython.utils.capture import capture_output
from .. import widget
from ..widget import Widget
from ..widget_button import Button
import copy
def test_no_widget_view():
    shell = InteractiveShell.instance()
    with capture_output() as cap:
        w = Widget()
        display(w)
    assert len(cap.outputs) == 1, 'expect 1 output'
    mime_bundle = cap.outputs[0].data
    assert mime_bundle['text/plain'] == repr(w), 'expected plain text output'
    assert 'application/vnd.jupyter.widget-view+json' not in mime_bundle, 'widget has no view'
    assert cap.stdout == '', repr(cap.stdout)
    assert cap.stderr == '', repr(cap.stderr)