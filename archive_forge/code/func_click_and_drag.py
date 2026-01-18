from unittest import mock
import matplotlib.pyplot as plt
def click_and_drag(tool, start, end, key=None):
    """
    Helper to simulate a mouse drag operation.

    Parameters
    ----------
    tool : `~matplotlib.widgets.Widget`
    start : [float, float]
        Starting point in data coordinates.
    end : [float, float]
        End point in data coordinates.
    key : None or str
         An optional key that is pressed during the whole operation
         (see also `.KeyEvent`).
    """
    if key is not None:
        do_event(tool, 'on_key_press', xdata=start[0], ydata=start[1], button=1, key=key)
    do_event(tool, 'press', xdata=start[0], ydata=start[1], button=1)
    do_event(tool, 'onmove', xdata=end[0], ydata=end[1], button=1)
    do_event(tool, 'release', xdata=end[0], ydata=end[1], button=1)
    if key is not None:
        do_event(tool, 'on_key_release', xdata=end[0], ydata=end[1], button=1, key=key)