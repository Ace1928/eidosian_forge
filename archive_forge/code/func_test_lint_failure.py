import ast
import sys
from pathlib import Path
import pytest
from trio._tests.pytest_plugin import skip_if_optional_else_raise
from trio._tools.gen_exports import (
from collections import Counter
from collections import Counter
from collections import Counter
import os
from typing import TYPE_CHECKING
@skip_lints
def test_lint_failure(tmp_path: Path) -> None:
    """Test that processing properly fails if black or ruff does."""
    try:
        import black
        import ruff
    except ImportError as error:
        skip_if_optional_else_raise(error)
    file = File(tmp_path / 'module.py', 'module')
    with pytest.raises(SystemExit):
        run_linters(file, 'class not valid code ><')
    with pytest.raises(SystemExit):
        run_linters(file, 'import waffle\n;import trio')