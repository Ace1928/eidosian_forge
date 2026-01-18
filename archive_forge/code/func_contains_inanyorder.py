from typing import MutableSequence, Optional, Sequence, TypeVar, Union, cast
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
def contains_inanyorder(*items: Union[Matcher[T], T]) -> Matcher[Sequence[T]]:
    """Matches if sequences's elements, in any order, satisfy a given list of
    matchers.

    :param match1,...: A comma-separated list of matchers.

    This matcher iterates the evaluated sequence, seeing if each element
    satisfies any of the given matchers. The matchers are tried from left to
    right, and when a satisfied matcher is found, it is no longer a candidate
    for the remaining elements. If a one-to-one correspondence is established
    between elements and matchers, ``contains_inanyorder`` is satisfied.

    Any argument that is not a matcher is implicitly wrapped in an
    :py:func:`~hamcrest.core.core.isequal.equal_to` matcher to check for
    equality.

    """
    matchers = []
    for item in items:
        matchers.append(wrap_matcher(item))
    return IsSequenceContainingInAnyOrder(matchers)