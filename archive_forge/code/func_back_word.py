from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on('<Esc+b>')
@edit_keys.on('<Ctrl-LEFT>')
@edit_keys.on('<Esc+LEFT>')
def back_word(cursor_offset, line):
    return (last_word_pos(line[:cursor_offset]), line)