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
def test_create_pass_through_args() -> None:
    testcases = [('def f()', '()'), ('def f(one)', '(one)'), ('def f(one, two)', '(one, two)'), ('def f(one, *args)', '(one, *args)'), ('def f(one, *args, kw1, kw2=None, **kwargs)', '(one, *args, kw1=kw1, kw2=kw2, **kwargs)')]
    for funcdef, expected in testcases:
        func_node = ast.parse(funcdef + ':\n  pass').body[0]
        assert isinstance(func_node, ast.FunctionDef)
        assert create_passthrough_args(func_node) == expected