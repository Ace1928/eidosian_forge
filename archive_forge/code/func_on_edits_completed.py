import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
def on_edits_completed(self, fn):
    """
        Register a function to be called after all pending trace and layout
        edit operations have completed

        If there are no pending edit operations then function is called
        immediately

        Parameters
        ----------
        fn : callable
            Function of zero arguments to be called when all pending edit
            operations have completed
        """
    if self._layout_edit_in_process or self._trace_edit_in_process:
        self._waiting_edit_callbacks.append(fn)
    else:
        fn()