import inspect
import warnings
import re
import os
import collections
from itertools import islice
from tokenize import open as open_py_source
from .logger import pformat
 Returns a nicely formatted statement displaying the function
        call with the given arguments.
    