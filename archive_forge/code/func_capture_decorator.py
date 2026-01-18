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
def capture_decorator(func):

    @wraps(func)
    def inner(*args, **kwargs):
        if clear_output:
            self.clear_output(*clear_args, **clear_kwargs)
        with self:
            return func(*args, **kwargs)
    return inner