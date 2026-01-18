import sys
from testtools import TestResult
from testtools.content import StackLinesContent
from testtools.matchers import (
from testtools import runtest
def hide_testtools_stack(should_hide=True):
    result = StackLinesContent.HIDE_INTERNAL_STACK
    StackLinesContent.HIDE_INTERNAL_STACK = should_hide
    return result