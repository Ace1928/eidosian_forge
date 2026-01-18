import os
import sys
from keras.src.api_export import keras_export
from keras.src.utils import io_utils
def check_graphviz():
    """Returns True if both PyDot and Graphviz are available."""
    if not check_pydot():
        return False
    try:
        pydot.Dot.create(pydot.Dot())
        return True
    except (OSError, pydot.InvocationException):
        return False