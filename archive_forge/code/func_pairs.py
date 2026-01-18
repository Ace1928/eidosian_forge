import itertools
import pytest
from referencing import Resource, exceptions
def pairs(choices):
    return itertools.combinations(choices, 2)