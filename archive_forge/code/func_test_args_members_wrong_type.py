import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_args_members_wrong_type(self):
    with pytest.raises(TypeError):
        Affine(0, 2, 3, None, None, '')