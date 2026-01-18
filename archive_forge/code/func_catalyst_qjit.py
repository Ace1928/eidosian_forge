from functools import wraps
from importlib.metadata import distribution
import warnings
import pennylane as qml
from .tape_mpl import tape_mpl
from .tape_text import tape_text
def catalyst_qjit(qnode):
    """The ``catalyst.while`` wrapper method"""
    try:
        distribution('pennylane_catalyst')
        return qnode.__class__.__name__ == 'QJIT'
    except ImportError:
        return False