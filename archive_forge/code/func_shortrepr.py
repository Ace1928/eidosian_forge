import sys
import os
import re
import warnings
import types
import unicodedata
def shortrepr(self):
    if self['names']:
        return '<%s "%s"...>' % (self.__class__.__name__, '; '.join([ensure_str(n) for n in self['names']]))
    else:
        return '<%s...>' % self.tagname