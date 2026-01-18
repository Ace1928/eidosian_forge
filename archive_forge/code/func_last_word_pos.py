from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
def last_word_pos(string):
    """returns the start index of the last word of given string"""
    match = forward_word_re.search(string[::-1])
    index = match and len(string) - match.end() + 1
    return index or 0