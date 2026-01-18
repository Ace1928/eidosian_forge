from typing import Any, List, Sequence, Tuple, TypeVar
from hamcrest import (
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.core.allof import AllOf
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
from typing_extensions import Protocol
from twisted.python.failure import Failure
def matches_result(successes: Matcher[Any]=equal_to(0), errors: Matcher[Any]=has_length(0), failures: Matcher[Any]=has_length(0), skips: Matcher[Any]=has_length(0), expectedFailures: Matcher[Any]=has_length(0), unexpectedSuccesses: Matcher[Any]=has_length(0)) -> Matcher[Any]:
    """
    Match a L{TestCase} instances with matching attributes.
    """
    return has_properties({'successes': successes, 'errors': errors, 'failures': failures, 'skips': skips, 'expectedFailures': expectedFailures, 'unexpectedSuccesses': unexpectedSuccesses})