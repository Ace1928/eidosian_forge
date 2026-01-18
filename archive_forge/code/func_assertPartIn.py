import os
import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
def assertPartIn(self, needle, haystack, message=''):
    self.assertTrue(any((needle in s for s in haystack)), message)