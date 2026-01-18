from __future__ import unicode_literals
import re
import pybtex.io
from pybtex.bibtex.interpreter import (
from pybtex.scanner import (
def process_string_literal(value):
    assert value.startswith('"')
    assert value.endswith('"')
    return String(value[1:-1])