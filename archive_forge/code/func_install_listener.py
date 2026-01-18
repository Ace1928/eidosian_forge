import os
import json
import atexit
import abc
import enum
import time
import threading
from timeit import default_timer as timer
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from numba.core import config
@contextmanager
def install_listener(kind, listener):
    """Install a listener for event "kind" temporarily within the duration of
    the context.

    Returns
    -------
    res : Listener
        The *listener* provided.

    Examples
    --------

    >>> with install_listener("numba:compile", listener):
    >>>     some_code()  # listener will be active here.
    >>> other_code()     # listener will be unregistered by this point.

    """
    register(kind, listener)
    try:
        yield listener
    finally:
        unregister(kind, listener)