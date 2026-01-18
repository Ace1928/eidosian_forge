import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def renderContents(self, encoding=DEFAULT_OUTPUT_ENCODING, prettyPrint=False, indentLevel=0):
    """Deprecated method for BS3 compatibility."""
    if not prettyPrint:
        indentLevel = None
    return self.encode_contents(indent_level=indentLevel, encoding=encoding)