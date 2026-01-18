from suds import *
from logging import getLogger
def print_collection(self, c, h, n):
    """Print collection using the specified indent (n) and newline (nl)."""
    if c in h:
        return '[]...'
    h.append(c)
    s = []
    for item in c:
        s.append('\n')
        s.append(self.indent(n))
        s.append(self.process(item, h, n - 2))
        s.append(',')
    h.pop()
    return ''.join(s)