import os
import sys
from functools import partial
from os.path import isabs
from typing import (
from unittest import TestCase, TestSuite
from attrs import define, field, frozen
from attrs.converters import default_if_none
from twisted.internet.defer import Deferred, DeferredList, gatherResults
from twisted.internet.interfaces import IReactorCore, IReactorProcess
from twisted.logger import Logger
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
from twisted.python.modules import theSystemPath
from .._asyncrunner import _iterateTests
from ..itrial import IReporter, ITestCase
from ..reporter import UncleanWarningsReporterWrapper
from ..runner import TestHolder
from ..util import _unusedTestDirectory, openTestLog
from . import _WORKER_AMP_STDIN, _WORKER_AMP_STDOUT
from .distreporter import DistReporter
from .functional import countingCalls, discardResult, iterateWhile, takeWhile
from .worker import LocalWorker, LocalWorkerAMP, WorkerAction
def shouldContinue(untilFailure: bool, result: IReporter) -> bool:
    """
    Determine whether the test suite should be iterated again.

    @param untilFailure: C{True} if the suite is supposed to run until
        failure.

    @param result: The test result of the test suite iteration which just
        completed.
    """
    return untilFailure and result.wasSuccessful()