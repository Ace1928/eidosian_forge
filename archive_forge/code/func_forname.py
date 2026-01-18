from collections import deque
from functools import reduce
from math import ceil, floor
import operator
import re
from itertools import chain
import six
from genshi.compat import IS_PYTHON2
from genshi.core import Stream, Attrs, Namespace, QName
from genshi.core import START, END, TEXT, START_NS, END_NS, COMMENT, PI, \
@classmethod
def forname(cls, name):
    """Return the axis constant for the given name, or `None` if no such
        axis was defined.
        """
    return getattr(cls, name.upper().replace('-', '_'), None)