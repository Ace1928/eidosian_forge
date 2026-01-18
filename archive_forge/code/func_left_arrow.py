from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on(config='left_key')
@edit_keys.on('<LEFT>')
def left_arrow(cursor_offset, line):
    return (max(0, cursor_offset - 1), line)