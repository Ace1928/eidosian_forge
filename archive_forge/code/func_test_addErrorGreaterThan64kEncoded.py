from __future__ import annotations
from typing import Sized
from unittest import TestCase
from hamcrest import assert_that, equal_to, has_length
from hamcrest.core.matcher import Matcher
from twisted.internet.defer import Deferred
from twisted.test.iosim import connectedServerAndClient
from twisted.trial._dist.worker import LocalWorkerAMP, WorkerProtocol
from twisted.trial.reporter import TestResult
from twisted.trial.test import erroneous, pyunitcases, sample, skipping
from twisted.trial.unittest import SynchronousTestCase
from .matchers import matches_result
def test_addErrorGreaterThan64kEncoded(self) -> None:
    """
        L{WorkerReporter} propagates errors with a string representation that
        is smaller than an implementation-specific limit but which encode to a
        byte representation that exceeds this limit.
        """
    self.assertTestRun(erroneous.TestAsynchronousFail('test_exceptionGreaterThan64kEncoded'), errors=has_length(1))