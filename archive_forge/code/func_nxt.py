from __future__ import unicode_literals
import re
def nxt(self):
    """ for backwards compatibility """
    try:
        cur, entering = next(self)
        return {'entering': entering, 'node': cur}
    except StopIteration:
        return None