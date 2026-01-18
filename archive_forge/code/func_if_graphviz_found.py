import subprocess
import sys
from collections import namedtuple
from io import StringIO
from subprocess import PIPE
from typing import Any, Callable, Dict, Generator, Optional, Tuple
import pytest
from sphinx.testing import util
from sphinx.testing.util import SphinxTestApp, SphinxTestAppWrapperForSkipBuilding
@pytest.fixture
def if_graphviz_found(app: SphinxTestApp) -> None:
    """
    The test will be skipped when using 'if_graphviz_found' fixture and graphviz
    dot command is not found.
    """
    graphviz_dot = getattr(app.config, 'graphviz_dot', '')
    try:
        if graphviz_dot:
            subprocess.run([graphviz_dot, '-V'], stdout=PIPE, stderr=PIPE)
            return
    except OSError:
        pass
    pytest.skip('graphviz "dot" is not available')