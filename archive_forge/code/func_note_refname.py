import sys
import os
import re
import warnings
import types
import unicodedata
def note_refname(self, node):
    self.refnames.setdefault(node['refname'], []).append(node)