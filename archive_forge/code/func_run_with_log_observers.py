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
def run_with_log_observers(observers, function, *args, **kwargs):
    """Run 'function' with the given Twisted log observers."""
    warnings.warn('run_with_log_observers is deprecated since 1.8.2.', DeprecationWarning, stacklevel=2)
    with _NoTwistedLogObservers():
        with _TwistedLogObservers(observers):
            return function(*args, **kwargs)