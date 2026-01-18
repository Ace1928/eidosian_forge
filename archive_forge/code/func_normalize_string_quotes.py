import re
import sys
from functools import lru_cache
from typing import Final, List, Match, Pattern
from black._width_table import WIDTH_TABLE
from blib2to3.pytree import Leaf
def normalize_string_quotes(s: str) -> str:
    """Prefer double quotes but only if it doesn't cause more escaping.

    Adds or removes backslashes as appropriate. Doesn't parse and fix
    strings nested in f-strings.
    """
    value = s.lstrip(STRING_PREFIX_CHARS)
    if value[:3] == '"""':
        return s
    elif value[:3] == "'''":
        orig_quote = "'''"
        new_quote = '"""'
    elif value[0] == '"':
        orig_quote = '"'
        new_quote = "'"
    else:
        orig_quote = "'"
        new_quote = '"'
    first_quote_pos = s.find(orig_quote)
    if first_quote_pos == -1:
        return s
    prefix = s[:first_quote_pos]
    unescaped_new_quote = _cached_compile(f'(([^\\\\]|^)(\\\\\\\\)*){new_quote}')
    escaped_new_quote = _cached_compile(f'([^\\\\]|^)\\\\((?:\\\\\\\\)*){new_quote}')
    escaped_orig_quote = _cached_compile(f'([^\\\\]|^)\\\\((?:\\\\\\\\)*){orig_quote}')
    body = s[first_quote_pos + len(orig_quote):-len(orig_quote)]
    if 'r' in prefix.casefold():
        if unescaped_new_quote.search(body):
            return s
        new_body = body
    else:
        new_body = sub_twice(escaped_new_quote, f'\\1\\2{new_quote}', body)
        if body != new_body:
            body = new_body
            s = f'{prefix}{orig_quote}{body}{orig_quote}'
        new_body = sub_twice(escaped_orig_quote, f'\\1\\2{orig_quote}', new_body)
        new_body = sub_twice(unescaped_new_quote, f'\\1\\\\{new_quote}', new_body)
    if 'f' in prefix.casefold():
        matches = re.findall('\n            (?:(?<!\\{)|^)\\{  # start of the string or a non-{ followed by a single {\n                ([^{].*?)  # contents of the brackets except if begins with {{\n            \\}(?:(?!\\})|$)  # A } followed by end of the string or a non-}\n            ', new_body, re.VERBOSE)
        for m in matches:
            if '\\' in str(m):
                return s
    if new_quote == '"""' and new_body[-1:] == '"':
        new_body = new_body[:-1] + '\\"'
    orig_escape_count = body.count('\\')
    new_escape_count = new_body.count('\\')
    if new_escape_count > orig_escape_count:
        return s
    if new_escape_count == orig_escape_count and orig_quote == '"':
        return s
    return f'{prefix}{new_quote}{new_body}{new_quote}'