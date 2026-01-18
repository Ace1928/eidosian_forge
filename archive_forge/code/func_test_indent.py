import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_indent(self):
    code = '\n'.join(argmap._indent(*['try:', 'try:', 'pass#', 'finally:', 'pass#', '#', 'finally:', 'pass#']))
    assert code == 'try:\n try:\n  pass#\n finally:\n  pass#\n #\nfinally:\n pass#'