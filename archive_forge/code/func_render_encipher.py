import os
import textwrap
from passlib.utils.compat import irange
def render_encipher(write, indent=0):
    for i in irange(0, 15, 2):
        write(indent, '            # Feistel substitution on left word (round %(i)d)\n            r ^= %(left)s ^ p%(i1)d\n\n            # Feistel substitution on right word (round %(i1)d)\n            l ^= %(right)s ^ p%(i2)d\n        ', i=i, i1=i + 1, i2=i + 2, left=BFSTR, right=BFSTR.replace('l', 'r'))