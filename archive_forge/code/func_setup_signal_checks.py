import threading
from shapely.lib import _setup_signal_checks, GEOSException, ShapelyError  # NOQA
def setup_signal_checks(interval=10000):
    """This enables Python signal checks in the ufunc inner loops.

    Doing so allows termination (using CTRL+C) of operations on large arrays of vectors.

    Parameters
    ----------
    interval : int, default 10000
        Check for interrupts every x iterations. The higher the number, the slower
        shapely will respond to a signal. However, at low values there will be a negative effect
        on performance. The default of 10000 does not have any measureable effects on performance.

    Notes
    -----
    For more information on signals consult the Python docs:

    https://docs.python.org/3/library/signal.html
    """
    if interval <= 0:
        raise ValueError('Signal checks interval must be greater than zero.')
    _setup_signal_checks(interval, threading.main_thread().ident)