from __future__ import unicode_literals
import re
import pybtex.io
from pybtex.bibtex.interpreter import (
from pybtex.scanner import (
def process_identifier(name):
    if name[0] == "'":
        return QuotedVar(name[1:])
    else:
        return Identifier(name)