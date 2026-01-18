import sys
import os
import re
import warnings
import types
import unicodedata
def note_refid(self, node):
    self.refids.setdefault(node['refid'], []).append(node)