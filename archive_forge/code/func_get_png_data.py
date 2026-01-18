from traitlets import (
from ipywidgets import DOMWidget, register, widget_serialization
from .scales import Scale, LinearScale
from .interacts import Interaction
from .marks import Mark
from .axes import Axis
from ._version import __frontend_version__
def get_png_data(self, callback, scale=None):
    """
        Gets the Figure as a PNG memory view

        Parameters
        ----------
        callback: callable
            Called with the PNG data as the only positional argument.

        scale: float (default: None)
            Scale up the png resolution when scale > 1, when not given base this on the screen pixel ratio.
        """
    if self._upload_png_callback:
        raise Exception('get_png_data already in progress')
    self._upload_png_callback = callback
    self.send({'type': 'upload_png', 'scale': scale})