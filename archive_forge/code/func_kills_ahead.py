from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
def kills_ahead(func):
    func.kills = 'ahead'
    return func