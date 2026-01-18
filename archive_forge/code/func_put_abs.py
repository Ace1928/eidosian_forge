import codecs
import copy
import sys
import warnings
def put_abs(self, r, c, ch):
    """Screen array starts at 1 index."""
    r = constrain(r, 1, self.rows)
    c = constrain(c, 1, self.cols)
    if isinstance(ch, bytes):
        ch = self._decode(ch)[0]
    else:
        ch = ch[0]
    self.w[r - 1][c - 1] = ch