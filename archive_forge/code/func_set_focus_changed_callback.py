from __future__ import annotations
import functools
import typing
import warnings
def set_focus_changed_callback(self, callback: Callable[[int], typing.Any]) -> None:
    """
        Assign a callback to be called when the focus index changes
        for any reason.  The callback is in the form:

        callback(new_focus)
        new_focus -- new focus index

        >>> import sys
        >>> ml = MonitoredFocusList([1,2,3], focus=1)
        >>> ml.set_focus_changed_callback(lambda f: sys.stdout.write("focus: %d\\n" % (f,)))
        >>> ml
        MonitoredFocusList([1, 2, 3], focus=1)
        >>> ml.append(10)
        >>> ml.insert(1, 11)
        focus: 2
        >>> ml
        MonitoredFocusList([1, 11, 2, 3, 10], focus=2)
        >>> del ml[:2]
        focus: 0
        >>> ml[:0] = [12, 13, 14]
        focus: 3
        >>> ml.focus = 5
        focus: 5
        >>> ml
        MonitoredFocusList([12, 13, 14, 2, 3, 10], focus=5)
        """
    self._focus_changed = callback