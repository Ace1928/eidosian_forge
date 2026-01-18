import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def bar_self_referential(self, *args, **kwargs):
    self.bar()