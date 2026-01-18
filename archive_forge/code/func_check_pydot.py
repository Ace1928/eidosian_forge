import os
import sys
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.util import nest
def check_pydot():
    """Returns True if PyDot and Graphviz are available."""
    if pydot is None:
        return False
    try:
        pydot.Dot.create(pydot.Dot())
        return True
    except (OSError, pydot.InvocationException):
        return False