import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
@classmethod
def bar_cls_args(cls, *args):
    return (cls,) + args