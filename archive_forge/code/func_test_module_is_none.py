import sys
import dill
import test_mixins as module
from importlib import reload
import os
import math
def test_module_is_none():
    assert obj.__module__ is None
    assert dill.copy(obj)(3) == obj(3)