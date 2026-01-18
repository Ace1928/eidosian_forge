import inspect
import logging
import warnings
import tensorflow as tf
from tensorflow.python.eager import context
import pennylane as qml
from pennylane.measurements import Shots
def set_parameters_on_copy(tapes, params):
    """Copy a set of tapes with operations and set parameters"""
    return tuple((t.bind_new_parameters(a, list(range(len(a)))) for t, a in zip(tapes, params)))