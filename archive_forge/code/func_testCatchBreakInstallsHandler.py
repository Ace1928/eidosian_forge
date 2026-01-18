import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def testCatchBreakInstallsHandler(self):
    module = sys.modules['unittest.main']
    original = module.installHandler

    def restore():
        module.installHandler = original
    self.addCleanup(restore)
    self.installed = False

    def fakeInstallHandler():
        self.installed = True
    module.installHandler = fakeInstallHandler
    program = self.program
    program.catchbreak = True
    program.testRunner = FakeRunner
    program.runTests()
    self.assertTrue(self.installed)