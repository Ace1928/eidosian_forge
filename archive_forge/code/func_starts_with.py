from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.matcher import Matcher
from hamcrest.library.text.substringmatcher import SubstringMatcher
def starts_with(substring: str) -> Matcher[str]:
    """Matches if object is a string starting with a given string.

    :param string: The string to search for.

    This matcher first checks whether the evaluated object is a string. If so,
    it checks if ``string`` matches the beginning characters of the evaluated
    object.

    Example::

        starts_with("foo")

    will match "foobar".

    """
    return StringStartsWith(substring)