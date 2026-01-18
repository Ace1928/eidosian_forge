import sys
import os
import re
import warnings
import types
import unicodedata
def note_footnote(self, footnote):
    self.set_id(footnote)
    self.footnotes.append(footnote)