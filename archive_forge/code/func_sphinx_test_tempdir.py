import subprocess
import sys
from collections import namedtuple
from io import StringIO
from subprocess import PIPE
from typing import Any, Callable, Dict, Generator, Optional, Tuple
import pytest
from sphinx.testing import util
from sphinx.testing.util import SphinxTestApp, SphinxTestAppWrapperForSkipBuilding
@pytest.fixture(scope='session')
def sphinx_test_tempdir(tmpdir_factory: Any) -> 'util.path':
    """
    Temporary directory wrapped with `path` class.
    """
    tmpdir = tmpdir_factory.getbasetemp()
    return util.path(tmpdir).abspath()