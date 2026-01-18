import errno
import os
import re
import sys
from inspect import getmro
from io import BytesIO, StringIO
from typing import Type
from unittest import (
from hamcrest import assert_that, equal_to, has_item, has_length
from twisted.python import log
from twisted.python.failure import Failure
from twisted.trial import itrial, reporter, runner, unittest, util
from twisted.trial.reporter import UncleanWarningsReporterWrapper, _ExitWrapper
from twisted.trial.test import erroneous, sample
from twisted.trial.unittest import SkipTest, Todo, makeTodo
from .._dist.test.matchers import isFailure, matches_result, similarFrame
from .matchers import after
def removeMethod(self, klass, methodName):
    """
        Remove 'methodName' from 'klass'.

        If 'klass' does not have a method named 'methodName', then
        'removeMethod' succeeds silently.

        If 'klass' does have a method named 'methodName', then it is removed
        using delattr. Also, methods of the same name are removed from all
        base classes of 'klass', thus removing the method entirely.

        @param klass: The class to remove the method from.
        @param methodName: The name of the method to remove.
        """
    method = getattr(klass, methodName, None)
    if method is None:
        return
    for base in getmro(klass):
        try:
            delattr(base, methodName)
        except (AttributeError, TypeError):
            break
        else:
            self.addCleanup(setattr, base, methodName, method)