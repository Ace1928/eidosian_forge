import sys
from functools import wraps
from .domwidget import DOMWidget
from .trait_types import TypedTuple
from .widget import register
from .._version import __jupyter_widgets_output_version__
from traitlets import Unicode, Dict
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output
from IPython import get_ipython
import traceback
def append_stdout(self, text):
    """Append text to the stdout stream."""
    self._append_stream_output(text, stream_name='stdout')