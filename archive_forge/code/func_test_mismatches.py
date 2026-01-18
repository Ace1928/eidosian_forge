from typing import Callable, Sequence, Tuple, Type
from hamcrest import anything, assert_that, contains, contains_string, equal_to, not_
from hamcrest.core.matcher import Matcher
from hamcrest.core.string_description import StringDescription
from hypothesis import given
from hypothesis.strategies import (
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .matchers import HasSum, IsSequenceOf, S, isFailure, similarFrame
@given(sampled_from([ValueError, ZeroDivisionError, RuntimeError]))
def test_mismatches(self, excType: Type[BaseException]) -> None:
    """
        L{isFailure} does not match instances of L{Failure} with
        attributes that don't match.

        :param excType: An exception type to wrap in a L{Failure} to be
            matched against.
        """
    matcher = isFailure(type=equal_to(excType), other=not_(anything()))
    failure = Failure(excType())
    assert_that(matcher.matches(failure), equal_to(False))