import functools
import re
import tokenize
from hacking import core
@core.flake8ext
def check_dict_formatting_in_string(logical_line, tokens):
    """Check that strings do not use dict-formatting with a single replacement

    N352
    """
    if not logical_line or logical_line.startswith('#') or logical_line.endswith('# noqa'):
        return
    current_string = ''
    in_string = False
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.STRING:
            if not in_string:
                current_string = ''
                in_string = True
            current_string += text.strip('"')
        elif token_type == tokenize.OP:
            if not current_string:
                continue
            in_string = False
            if text == '%':
                format_keys = set()
                for match in re_str_format.finditer(current_string):
                    format_keys.add(match.group(1))
                if len(format_keys) == 1:
                    yield (0, 'N352 Do not use mapping key string formatting with a single key')
            if text != ')':
                current_string = ''
        elif token_type in (tokenize.NL, tokenize.COMMENT):
            continue
        else:
            in_string = False
            if token_type == tokenize.NEWLINE:
                current_string = ''