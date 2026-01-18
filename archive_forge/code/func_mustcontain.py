import os
from fnmatch import fnmatch
from datetime import datetime
import operator
import re
def mustcontain(self, s):
    __tracebackhide__ = True
    bytes = self.bytes
    if s not in bytes:
        print('Could not find %r in:' % s)
        print(bytes)
        assert s in bytes