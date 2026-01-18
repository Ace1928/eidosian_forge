import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def no_cycle():
    b = []
    b.append([])
    return b