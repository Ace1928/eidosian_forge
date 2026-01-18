import io
import warnings
import sys
from fixtures import CompoundFixture, Fixture
from testtools.content import Content, text_content
from testtools.content_type import UTF8_TEXT
from testtools.runtest import RunTest, _raise_force_fail_error
from ._deferred import extract_result
from ._spinner import (
from twisted.internet import defer
from twisted.python import log
def assert_fails_with(d, *exc_types, **kwargs):
    """Assert that ``d`` will fail with one of ``exc_types``.

    The normal way to use this is to return the result of
    ``assert_fails_with`` from your unit test.

    Equivalent to Twisted's ``assertFailure``.

    :param Deferred d: A ``Deferred`` that is expected to fail.
    :param exc_types: The exception types that the Deferred is expected to
        fail with.
    :param type failureException: An optional keyword argument.  If provided,
        will raise that exception instead of
        ``testtools.TestCase.failureException``.
    :return: A ``Deferred`` that will fail with an ``AssertionError`` if ``d``
        does not fail with one of the exception types.
    """
    failureException = kwargs.pop('failureException', None)
    if failureException is None:
        from testtools import TestCase
        failureException = TestCase.failureException
    expected_names = ', '.join((exc_type.__name__ for exc_type in exc_types))

    def got_success(result):
        raise failureException(f'{expected_names} not raised ({result!r} returned)')

    def got_failure(failure):
        if failure.check(*exc_types):
            return failure.value
        raise failureException('{} raised instead of {}:\n {}'.format(failure.type.__name__, expected_names, failure.getTraceback()))
    return d.addCallbacks(got_success, got_failure)