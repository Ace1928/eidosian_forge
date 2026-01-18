import re
from typing import Any, Optional, Tuple
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
def described_as(description: str, matcher: Matcher[Any], *values) -> Matcher[Any]:
    """Adds custom failure description to a given matcher.

    :param description: Overrides the matcher's description.
    :param matcher: The matcher to satisfy.
    :param value1,...: Optional comma-separated list of substitution values.

    The description may contain substitution placeholders %0, %1, etc. These
    will be replaced by any values that follow the matcher.

    """
    return DescribedAs(description, matcher, *values)