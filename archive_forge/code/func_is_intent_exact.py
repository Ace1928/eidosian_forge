import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def is_intent_exact(self, *names):
    return len(self.intent_list) == len(names) and self.is_intent(*names)