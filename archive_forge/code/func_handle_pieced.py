from __future__ import print_function
import re
import hashlib
def handle_pieced(self, lines, spec):
    """Digest stuff according to the spec."""
    for offset, length in spec:
        for i in xrange(length):
            try:
                line = lines[int(offset * len(lines) // 100) + i]
            except IndexError:
                pass
            else:
                self.handle_line(line)