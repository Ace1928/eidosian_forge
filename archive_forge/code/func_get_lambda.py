import sys
import dill
import test_mixins as module
from importlib import reload
import os
import math
def get_lambda(str, **kwarg):
    return eval(str, kwarg, None)