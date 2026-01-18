import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_mul_sequence_wrong_member_types(self):

    class NotPtSeq:

        @classmethod
        def from_points(cls, points):
            list(points)

        def __iter__(self):
            yield 0
    with pytest.raises(TypeError):
        Affine(1, 2, 3, 4, 5, 6) * NotPtSeq()