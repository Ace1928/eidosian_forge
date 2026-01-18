import sys
from unittest import TestCase
from contextlib import contextmanager
from IPython.display import Markdown, Image
from ipywidgets import widget_output
@widget.capture(clear_output=False)
def captee(*args, **kwargs):
    assert widget.msg_id == msg_id