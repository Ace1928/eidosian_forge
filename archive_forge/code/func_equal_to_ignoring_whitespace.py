from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
def equal_to_ignoring_whitespace(string: str) -> Matcher[str]:
    """Matches if object is a string equal to a given string, ignoring
    differences in whitespace.

    :param string: The string to compare against as the expected value.

    This matcher first checks whether the evaluated object is a string. If so,
    it compares it with ``string``, ignoring differences in runs of whitespace.

    Example::

        equal_to_ignoring_whitespace("hello world")

    will match ``"hello   world"``.

    """
    return IsEqualIgnoringWhiteSpace(string)