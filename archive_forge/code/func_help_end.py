import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
@StatelessInputTransformer.wrap
def help_end(line: str):
    """Translate lines with ?/?? at the end"""
    m = _help_end_re.search(line)
    if m is None or ends_in_comment_or_string(line):
        return line
    target = m.group(1)
    esc = m.group(3)
    match = _initial_space_re.match(line)
    assert match is not None
    lspace = match.group(0)
    return _make_help_call(target, esc, lspace)