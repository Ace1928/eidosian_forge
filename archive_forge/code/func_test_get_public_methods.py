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
def test_get_public_methods() -> None:
    methods = list(get_public_methods(ast.parse(SOURCE)))
    assert {m.name for m in methods} == {'public_func', 'public_async_func'}