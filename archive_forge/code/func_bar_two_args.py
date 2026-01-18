import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def bar_two_args(self, arg=None):
    return (self, arg)