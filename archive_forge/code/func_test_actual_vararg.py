import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_actual_vararg(self):

    @argmap(lambda x: -x, 4)
    def foo(x, y, *args):
        return (x, y) + tuple(args)
    assert foo(1, 2, 3, 4, 5, 6) == (1, 2, 3, 4, -5, 6)