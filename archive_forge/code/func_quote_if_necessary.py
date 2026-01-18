import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def quote_if_necessary(s):
    """Enclose attribute value in quotes, if needed."""
    if isinstance(s, bool):
        if s is True:
            return 'True'
        return 'False'
    if not isinstance(s, str):
        return s
    if not s:
        return s
    if needs_quotes(s):
        replace = {'"': '\\"', '\n': '\\n', '\r': '\\r'}
        for a, b in replace.items():
            s = s.replace(a, b)
        return '"' + s + '"'
    return s