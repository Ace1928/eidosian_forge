import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
@add_two_to_second
@add_one_to_first_bad_decorator
def add_one_and_two(a, b):
    return (a, b)