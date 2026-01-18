from __future__ import annotations
import abc
import enum
import typing
import warnings
from .constants import Sizing, WHSettings
def set_focus_path(self, positions: Iterable[int | str]) -> None:
    """
        Set the .focus_position property starting from this container
        widget and proceeding along newly focused child widgets.  Any
        failed assignment due do incompatible position types or invalid
        positions will raise an IndexError.

        This method may be used to restore a particular widget to the
        focus by passing in the value returned from an earlier call to
        get_focus_path().

        positions -- sequence of positions
        """
    w: Widget = self
    for p in positions:
        if p != w.focus_position:
            w.focus_position = p
        w = w.focus.base_widget