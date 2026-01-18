import string
from twisted.logger import Logger
def stripEscapes(self, string):
    """
        Remove all ANSI color escapes from the given string.
        """
    result = ''
    show = 1
    i = 0
    L = len(string)
    while i < L:
        if show == 0 and string[i] in _sets:
            show = 1
        elif show:
            n = string.find('\x1b', i)
            if n == -1:
                return result + string[i:]
            else:
                result = result + string[i:n]
                i = n
                show = 0
        i = i + 1
    return result