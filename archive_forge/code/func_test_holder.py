import os
import pdb
import sys
import unittest as pyunit
from io import StringIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import plugin
from twisted.internet import defer
from twisted.plugins import twisted_trial
from twisted.python import failure, log, reflect
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedAny
from twisted.scripts import trial
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import _ForceGarbageCollectionDecorator
from twisted.trial.itrial import IReporter, ITestCase
def test_holder(self):
    """
        Check that L{runner.TestHolder} takes a description as a parameter
        and that this description is returned by the C{id} and
        C{shortDescription} methods.
        """
    self.assertEqual(self.holder.id(), self.description)
    self.assertEqual(self.holder.shortDescription(), self.description)