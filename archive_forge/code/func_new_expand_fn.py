import functools
import os
import copy
import warnings
import types
from typing import Sequence
import pennylane as qml
from pennylane.typing import ResultBatch
@new_dev.custom_expand
def new_expand_fn(self, tape, *args, **kwargs):
    tapes, _ = transform(tape, *targs, **tkwargs)
    tape = tapes[0]
    return self.default_expand_fn(tape, *args, **kwargs)