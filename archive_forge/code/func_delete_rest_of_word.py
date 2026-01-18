from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on('<Esc+d>')
@kills_ahead
def delete_rest_of_word(cursor_offset, line):
    m = delete_rest_of_word_re.search(line[cursor_offset:])
    if not m:
        return (cursor_offset, line, '')
    return (cursor_offset, line[:cursor_offset] + line[m.start() + cursor_offset + 1:], line[cursor_offset:m.start() + cursor_offset + 1])