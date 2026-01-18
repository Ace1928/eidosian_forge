import collections
import re
from humanfriendly.compat import coerce_string
from humanfriendly.terminal import (
def highlight_column_name(name):
    return ansi_wrap(name, bold=True, color=HIGHLIGHT_COLOR)