import re
from typing import Pattern, Union
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
def matches_regexp(pattern: Union[str, Pattern[str]]) -> Matcher[str]:
    """Matches if object is a string containing a match for a given regular
    expression.

    :param pattern: The regular expression to search for.

    This matcher first checks whether the evaluated object is a string. If so,
    it checks if the regular expression ``pattern`` matches anywhere within the
    evaluated object.

    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    return StringMatchesPattern(pattern)