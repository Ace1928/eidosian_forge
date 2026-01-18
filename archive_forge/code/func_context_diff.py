from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def context_diff(a, b, fromfile='', tofile='', fromfiledate='', tofiledate='', n=3, lineterm='\n'):
    """
    Compare two sequences of lines; generate the delta as a context diff.

    Context diffs are a compact way of showing line changes and a few
    lines of context.  The number of context lines is set by 'n' which
    defaults to three.

    By default, the diff control lines (those with *** or ---) are
    created with a trailing newline.  This is helpful so that inputs
    created from file.readlines() result in diffs that are suitable for
    file.writelines() since both the inputs and outputs have trailing
    newlines.

    For inputs that do not have trailing newlines, set the lineterm
    argument to "" so that the output will be uniformly newline free.

    The context diff format normally has a header for filenames and
    modification times.  Any or all of these may be specified using
    strings for 'fromfile', 'tofile', 'fromfiledate', and 'tofiledate'.
    The modification times are normally expressed in the ISO 8601 format.
    If not specified, the strings default to blanks.

    Example:

    >>> print(''.join(context_diff('one\\ntwo\\nthree\\nfour\\n'.splitlines(True),
    ...       'zero\\none\\ntree\\nfour\\n'.splitlines(True), 'Original', 'Current')),
    ...       end="")
    *** Original
    --- Current
    ***************
    *** 1,4 ****
      one
    ! two
    ! three
      four
    --- 1,4 ----
    + zero
      one
    ! tree
      four
    """
    _check_types(a, b, fromfile, tofile, fromfiledate, tofiledate, lineterm)
    prefix = dict(insert='+ ', delete='- ', replace='! ', equal='  ')
    started = False
    for group in SequenceMatcher(None, a, b).get_grouped_opcodes(n):
        if not started:
            started = True
            fromdate = '\t{}'.format(fromfiledate) if fromfiledate else ''
            todate = '\t{}'.format(tofiledate) if tofiledate else ''
            yield '*** {}{}{}'.format(fromfile, fromdate, lineterm)
            yield '--- {}{}{}'.format(tofile, todate, lineterm)
        first, last = (group[0], group[-1])
        yield ('***************' + lineterm)
        file1_range = _format_range_context(first[1], last[2])
        yield '*** {} ****{}'.format(file1_range, lineterm)
        if any((tag in {'replace', 'delete'} for tag, _, _, _, _ in group)):
            for tag, i1, i2, _, _ in group:
                if tag != 'insert':
                    for line in a[i1:i2]:
                        yield (prefix[tag] + line)
        file2_range = _format_range_context(first[3], last[4])
        yield '--- {} ----{}'.format(file2_range, lineterm)
        if any((tag in {'replace', 'insert'} for tag, _, _, _, _ in group)):
            for tag, _, _, j1, j2 in group:
                if tag != 'delete':
                    for line in b[j1:j2]:
                        yield (prefix[tag] + line)