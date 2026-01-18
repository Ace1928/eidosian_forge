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
def assertTestRun(self, target: TestCase, **expectations: Matcher[Sized]) -> None:
    """
        Run the given test and assert that the result matches the given
        expectations.
        """
    assert_that(run(self, target), matches_result(**expectations))