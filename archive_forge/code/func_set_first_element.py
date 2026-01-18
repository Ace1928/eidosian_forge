import os
import sys
from inspect import cleandoc
from itertools import chain
from string import ascii_letters, digits
from unittest import mock
import numpy as np
import pytest
import shapely
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def set_first_element(value, *args, **kwargs):
    for arg in chain(args, kwargs.values()):
        if hasattr(arg, '__setitem__'):
            arg[0] = value
            return arg