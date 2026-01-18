import sys, os
import textwrap
def set_usage(self, usage):
    if usage is None:
        self.usage = _('%prog [options]')
    elif usage is SUPPRESS_USAGE:
        self.usage = None
    elif usage.lower().startswith('usage: '):
        self.usage = usage[7:]
    else:
        self.usage = usage