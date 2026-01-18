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
@classmethod
def make_factory(cls, reactor=None, timeout=0.005, debug=False, suppress_twisted_logging=True, store_twisted_logs=True):
    """Make a factory that conforms to the RunTest factory interface.

        Example::

            class SomeTests(TestCase):
                # Timeout tests after two minutes.
                run_tests_with = AsynchronousDeferredRunTest.make_factory(
                    timeout=120)
        """

    class AsynchronousDeferredRunTestFactory:

        def __call__(self, case, handlers=None, last_resort=None):
            return cls(case, handlers, last_resort, reactor, timeout, debug, suppress_twisted_logging, store_twisted_logs)
    return AsynchronousDeferredRunTestFactory()