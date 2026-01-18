from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on(config='right_key')
@edit_keys.on('<RIGHT>')
def right_arrow(cursor_offset, line):
    return (min(len(line), cursor_offset + 1), line)