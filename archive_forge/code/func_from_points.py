import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
@classmethod
def from_points(cls, points):
    list(points)