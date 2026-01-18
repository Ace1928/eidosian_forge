from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on(config='clear_word_key')
@kills_behind
def delete_word_to_cursor(cursor_offset, line):
    start = 0
    for match in delete_word_to_cursor_re.finditer(line[:cursor_offset]):
        start = match.start() + 1
    return (start, line[:start] + line[cursor_offset:], line[start:cursor_offset])