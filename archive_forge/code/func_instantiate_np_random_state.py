import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
@np_random_state(1)
def instantiate_np_random_state(self, random_state):
    assert isinstance(random_state, np.random.RandomState)
    return random_state.random_sample()