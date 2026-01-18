from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on(config='clear_line_key')
def delete_from_cursor_back(cursor_offset, line):
    return (0, line[cursor_offset:])