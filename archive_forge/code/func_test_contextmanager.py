import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_contextmanager(self):
    container = []

    def contextmanager(x):
        nonlocal container
        return (x, lambda: container.append(x))

    @argmap(contextmanager, 0, 1, 2, try_finally=True)
    def foo(x, y, z):
        return (x, y, z)
    x, y, z = foo('a', 'b', 'c')
    assert container == ['c', 'b', 'a']