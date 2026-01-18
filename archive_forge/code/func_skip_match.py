import functools
import re
import sys
from Xlib.support import lock
def skip_match(self, complen):
    if len(self.path) + 1 >= complen:
        return None
    if self.skip:
        if self.db:
            return _Match(self.path + (MATCH_SKIP,), self.db)
        else:
            return None
    elif self.group[1]:
        return _Match(self.path + (MATCH_SKIP,), self.group[1])
    else:
        return None