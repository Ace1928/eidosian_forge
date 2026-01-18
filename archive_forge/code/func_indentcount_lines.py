import sys
def indentcount_lines(lines):
    """ Minimum indent for all lines in line list

    >>> lines = [' one', '  two', '   three']
    >>> indentcount_lines(lines)
    1
    >>> lines = []
    >>> indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> indentcount_lines(lines)
    1
    >>> indentcount_lines(['    '])
    0
    """
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return indentno