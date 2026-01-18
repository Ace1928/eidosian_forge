import os
import re
import shutil
import sys
from pathlib import Path
import pytest
@pytest.mark.skipif(mypy.__file__.endswith('.py'), reason='Non-compiled mypy is too slow')
def test_generation_is_disabled():
    """
    Makes sure we don't accidentally leave generation on
    """
    assert not GENERATE