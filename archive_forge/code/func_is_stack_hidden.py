import sys
from testtools import TestResult
from testtools.content import StackLinesContent
from testtools.matchers import (
from testtools import runtest
def is_stack_hidden():
    return StackLinesContent.HIDE_INTERNAL_STACK