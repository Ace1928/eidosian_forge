from unittest import mock
import matplotlib.pyplot as plt
def mock_event(ax, button=1, xdata=0, ydata=0, key=None, step=1):
    """
    Create a mock event that can stand in for `.Event` and its subclasses.

    This event is intended to be used in tests where it can be passed into
    event handling functions.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The axes the event will be in.
    xdata : float
        x coord of mouse in data coords.
    ydata : float
        y coord of mouse in data coords.
    button : None or `MouseButton` or {'up', 'down'}
        The mouse button pressed in this event (see also `.MouseEvent`).
    key : None or str
        The key pressed when the mouse event triggered (see also `.KeyEvent`).
    step : int
        Number of scroll steps (positive for 'up', negative for 'down').

    Returns
    -------
    event
        A `.Event`\\-like Mock instance.
    """
    event = mock.Mock()
    event.button = button
    event.x, event.y = ax.transData.transform([(xdata, ydata), (xdata, ydata)])[0]
    event.xdata, event.ydata = (xdata, ydata)
    event.inaxes = ax
    event.canvas = ax.figure.canvas
    event.key = key
    event.step = step
    event.guiEvent = None
    event.name = 'Custom'
    return event