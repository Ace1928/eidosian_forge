import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def add_one_to_first_bad_decorator(f):
    """Bad because it doesn't wrap the f signature (clobbers it)"""

    def decorated(a, *args, **kwargs):
        return f(a + 1, *args, **kwargs)
    return decorated