import sys, os
import textwrap
def set_short_opt_delimiter(self, delim):
    if delim not in ('', ' '):
        raise ValueError('invalid metavar delimiter for short options: %r' % delim)
    self._short_opt_fmt = '%s' + delim + '%s'