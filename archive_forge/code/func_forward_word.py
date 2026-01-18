from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on('<Esc+f>')
@edit_keys.on('<Ctrl-RIGHT>')
@edit_keys.on('<Esc+RIGHT>')
def forward_word(cursor_offset, line):
    match = forward_word_re.search(line[cursor_offset:] + ' ')
    delta = match.end() - 1 if match else 0
    return (cursor_offset + delta, line)