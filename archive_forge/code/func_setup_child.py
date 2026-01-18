import sys
import os
import re
import warnings
import types
import unicodedata
def setup_child(self, child):
    child.parent = self
    if self.document:
        child.document = self.document
        if child.source is None:
            child.source = self.document.current_source
        if child.line is None:
            child.line = self.document.current_line