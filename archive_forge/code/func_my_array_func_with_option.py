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
@implements(func_with_option)
def my_array_func_with_option(array, new_option='myarray'):
    return new_option