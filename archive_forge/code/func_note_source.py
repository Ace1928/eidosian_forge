import sys
import os
import re
import warnings
import types
import unicodedata
def note_source(self, source, offset):
    self.current_source = source
    if offset is None:
        self.current_line = offset
    else:
        self.current_line = offset + 1