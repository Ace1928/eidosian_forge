import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def format_list(extracted_list):
    """Format a list of tuples or FrameSummary objects for printing.

    Given a list of tuples or FrameSummary objects as returned by
    extract_tb() or extract_stack(), return a list of strings ready
    for printing.

    Each string in the resulting list corresponds to the item with the
    same index in the argument list.  Each string ends in a newline;
    the strings may contain internal newlines as well, for those items
    whose source text line is not None.
    """
    return StackSummary.from_list(extracted_list).format()