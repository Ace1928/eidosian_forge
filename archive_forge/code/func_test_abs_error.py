import builtins
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.expressions.expression import (
from cvxpy.tests.base_test import BaseTest
def test_abs_error(self) -> None:
    with pytest.raises(TypeError, match=__ABS_ERROR__):
        builtins.abs(self.x)