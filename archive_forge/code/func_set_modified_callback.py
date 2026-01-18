from __future__ import annotations
import functools
import typing
import warnings
def set_modified_callback(self, callback: Callable[[], typing.Any]) -> None:
    """
        Assign a callback function with no parameters that is called any
        time the list is modified.  Callback's return value is ignored.

        >>> import sys
        >>> ml = MonitoredList([1,2,3])
        >>> ml.set_modified_callback(lambda: sys.stdout.write("modified\\n"))
        >>> ml
        MonitoredList([1, 2, 3])
        >>> ml.append(10)
        modified
        >>> len(ml)
        4
        >>> ml += [11, 12, 13]
        modified
        >>> ml[:] = ml[:2] + ml[-2:]
        modified
        >>> ml
        MonitoredList([1, 2, 12, 13])
        """
    self._modified = callback