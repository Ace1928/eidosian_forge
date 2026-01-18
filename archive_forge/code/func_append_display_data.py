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
def append_display_data(self, display_object):
    """Append a display object as an output.

        Parameters
        ----------
        display_object : IPython.core.display.DisplayObject
            The object to display (e.g., an instance of
            `IPython.display.Markdown` or `IPython.display.Image`).
        """
    fmt = InteractiveShell.instance().display_formatter.format
    data, metadata = fmt(display_object)
    self.outputs += ({'output_type': 'display_data', 'data': data, 'metadata': metadata},)