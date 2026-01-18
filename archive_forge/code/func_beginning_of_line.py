from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on(config='beginning_of_line_key')
@edit_keys.on('<HOME>')
def beginning_of_line(cursor_offset, line):
    return (0, line)