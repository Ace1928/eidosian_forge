import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def extended_frame_gen():
    for f, lineno in frame_gen:
        yield (f, (lineno, None, None, None))