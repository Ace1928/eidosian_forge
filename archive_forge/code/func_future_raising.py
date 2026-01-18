import sys
import re
import asyncio
from typing import (
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
def future_raising(exception: Type[Exception], pattern=None, matching=None) -> Matcher[asyncio.Future]:
    """Matches a future with the expected exception.

    :param exception:  The class of the expected exception
    :param pattern:    Optional regular expression to match exception message.
    :param matching:   Optional Hamcrest matchers to apply to the exception.

    Expects the actual to be an already resolved future. The :py:func:`~hamcrest:core.core.future.resolved` helper can be used to wait for a future to resolve.
    Optional argument pattern should be a string containing a regular expression.  If provided,
    the string representation of the actual exception - e.g. `str(actual)` - must match pattern.

    Examples::

        assert_that(somefuture, future_exception(ValueError))
        assert_that(
            await resolved(async_http_get()),
            future_exception(HTTPError, matching=has_properties(status_code=500)
        )
    """
    return FutureRaising(exception, pattern, matching)