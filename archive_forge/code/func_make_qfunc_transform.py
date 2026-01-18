from copy import deepcopy
import functools
import inspect
import os
import warnings
import pennylane as qml
from pennylane.tape import make_qscript
@functools.wraps(tape_transform)
def make_qfunc_transform(fn):
    return _create_qfunc_internal_wrapper(fn, tape_transform, [], {})