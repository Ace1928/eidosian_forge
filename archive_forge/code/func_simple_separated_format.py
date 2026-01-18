from collections import namedtuple
from collections.abc import Iterable, Sized
from html import escape as htmlescape
from itertools import chain, zip_longest as izip_longest
from functools import reduce, partial
import io
import re
import math
import textwrap
import dataclasses
def simple_separated_format(separator):
    """Construct a simple TableFormat with columns separated by a separator.

    >>> tsv = simple_separated_format("\\t") ;         tabulate([["foo", 1], ["spam", 23]], tablefmt=tsv) == 'foo \\t 1\\nspam\\t23'
    True

    """
    return TableFormat(None, None, None, None, headerrow=DataRow('', separator, ''), datarow=DataRow('', separator, ''), padding=0, with_header_hide=None)