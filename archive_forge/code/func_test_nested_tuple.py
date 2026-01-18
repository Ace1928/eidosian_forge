import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_nested_tuple(self):

    def xform(x, y):
        u, v = y
        return (x + u + v, (x + u, x + v))

    @argmap(xform, (0, ('t', 2)))
    def foo(a, *args, **kwargs):
        return (a, args, kwargs)
    a, args, kwargs = foo(1, 2, 3, t=4)
    assert a == 1 + 4 + 3
    assert args == (2, 1 + 3)
    assert kwargs == {'t': 1 + 4}