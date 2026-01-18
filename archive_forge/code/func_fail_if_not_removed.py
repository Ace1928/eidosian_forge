import collections
import functools
import textwrap
import warnings
from packaging import version
from datetime import date
def fail_if_not_removed(method):
    """Decorate a test method to track removal of deprecated code

    This decorator catches :class:`~deprecation.UnsupportedWarning`
    warnings that occur during testing and causes unittests to fail,
    making it easier to keep track of when code should be removed.

    :raises: :class:`AssertionError` if an
             :class:`~deprecation.UnsupportedWarning`
             is raised while running the test method.
    """

    @functools.wraps(method)
    def test_inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            rv = method(*args, **kwargs)
        for warning in caught_warnings:
            if warning.category == UnsupportedWarning:
                raise AssertionError('%s uses a function that should be removed: %s' % (method, str(warning.message)))
        return rv
    return test_inner