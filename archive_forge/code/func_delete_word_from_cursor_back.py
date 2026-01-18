from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on('<Esc+BACKSPACE>')
@edit_keys.on('<Meta-BACKSPACE>')
@kills_behind
def delete_word_from_cursor_back(cursor_offset, line):
    """Whatever my option-delete does in bash on my mac"""
    if not line:
        return (cursor_offset, line, '')
    start = None
    for match in delete_word_from_cursor_back_re.finditer(line):
        if match.start() < cursor_offset:
            start = match.start()
    if start is not None:
        return (start, line[:start] + line[cursor_offset:], line[start:cursor_offset])
    else:
        return (cursor_offset, line, '')