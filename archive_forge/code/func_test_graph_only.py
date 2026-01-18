import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
@not_implemented_for('directed')
@not_implemented_for('multigraph')
def test_graph_only(G):
    pass