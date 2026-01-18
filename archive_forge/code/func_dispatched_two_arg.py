import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
@array_function_dispatch(lambda array1, array2: (array1, array2))
def dispatched_two_arg(array1, array2):
    """Docstring."""
    return 'original'