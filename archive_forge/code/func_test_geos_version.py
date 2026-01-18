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
def test_geos_version():
    expected = '{}.{}.{}'.format(*shapely.geos_version)
    actual = shapely.geos_version_string
    if any((c.isalpha() for c in actual)):
        if actual[-1].isnumeric():
            actual = actual.rstrip(digits)
        actual = actual.rstrip(ascii_letters)
    assert actual == expected