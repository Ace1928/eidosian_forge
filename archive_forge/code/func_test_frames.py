from typing import Callable, Sequence, Tuple, Type
from hamcrest import anything, assert_that, contains, contains_string, equal_to, not_
from hamcrest.core.matcher import Matcher
from hamcrest.core.string_description import StringDescription
from hypothesis import given
from hypothesis.strategies import (
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .matchers import HasSum, IsSequenceOf, S, isFailure, similarFrame
def test_frames(self):
    """
        The L{similarFrame} matcher matches elements of the C{frames} list
        of a L{Failure}.
        """
    try:
        raise ValueError('Oh no')
    except BaseException:
        f = Failure()
    actualDescription = StringDescription()
    matcher = isFailure(frames=contains(similarFrame('test_frames', 'test_matchers')))
    assert_that(matcher.matches(f, actualDescription), equal_to(True), actualDescription)