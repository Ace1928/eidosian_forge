import os
import re
import fnmatch
def glob2(dirname, pattern):
    assert _isrecursive(pattern)
    yield pattern[:0]
    for x in _rlistdir(dirname):
        yield x