from decimal import Decimal
from math import fabs
from typing import Any, Union, overload
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
def close_to(value, delta):
    """Matches if object is a number close to a given value, within a given
    delta.

    :param value: The value to compare against as the expected value.
    :param delta: The maximum delta between the values for which the numbers
        are considered close.

    This matcher compares the evaluated object against ``value`` to see if the
    difference is within a positive ``delta``.

    Example::

        close_to(3.0, 0.25)

    """
    return IsCloseTo(value, delta)