from __future__ import annotations
import typing
import warnings
from .attr_map import AttrMap
def set_w(self, new_widget: Widget) -> None:
    warnings.warn('backwards compatibility, widget used to be stored as original_widget', DeprecationWarning, stacklevel=2)
    self.original_widget = new_widget