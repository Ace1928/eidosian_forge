import sys, os
import textwrap
def ensure_value(self, attr, value):
    if not hasattr(self, attr) or getattr(self, attr) is None:
        setattr(self, attr, value)
    return getattr(self, attr)