import contextlib
import re
import sys
def expect_raises_with_proper_context(except_cls, check_context=True):
    return _expect_raises(except_cls, check_context=check_context)