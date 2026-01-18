import re, os, stat, io
from xdg.Exceptions import (ParsingError, DuplicateGroupError, NoGroupError,
import xdg.Locale
from xdg.util import u
def removeGroup(self, group):
    existed = group in self.content
    if existed:
        del self.content[group]
        self.tainted = True
    elif debug:
        raise NoGroupError(group, self.filename)
    return existed