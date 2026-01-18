from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on(config='cut_to_buffer_key')
@kills_ahead
def delete_from_cursor_forward(cursor_offset, line):
    return (cursor_offset, line[:cursor_offset], line[cursor_offset:])