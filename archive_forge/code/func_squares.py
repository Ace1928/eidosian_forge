import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def squares(n):
    """
    Generic iterable without a valid len, for testing purposes.

    Parameters
    ----------
    n : int
        Limit for computation.

    Returns
    -------
    squares : generator
        Generator yielding the first n squares.
    """
    return (x * x for x in range(n))