import sys
from testtools import TestResult
from testtools.content import StackLinesContent
from testtools.matchers import (
from testtools import runtest
def run_with_stack_hidden(should_hide, f, *args, **kwargs):
    old_should_hide = hide_testtools_stack(should_hide)
    try:
        return f(*args, **kwargs)
    finally:
        hide_testtools_stack(old_should_hide)