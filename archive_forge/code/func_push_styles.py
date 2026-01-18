import re
from humanfriendly.compat import HTMLParser, StringIO, name2codepoint, unichr
from humanfriendly.text import compact_empty_lines
from humanfriendly.terminal import ANSI_COLOR_CODES, ANSI_RESET, ansi_style
def push_styles(self, **changes):
    """
        Push new style information onto the stack.

        :param changes: Any keyword arguments are passed on to :func:`.ansi_style()`.

        This method is a helper for :func:`handle_starttag()`
        that does the following:

        1. Make a copy of the current styles (from the top of the stack),
        2. Apply the given `changes` to the copy of the current styles,
        3. Add the new styles to the stack,
        4. Emit the appropriate ANSI escape sequence to the output stream.
        """
    prototype = self.current_style
    if prototype:
        new_style = dict(prototype)
        new_style.update(changes)
    else:
        new_style = changes
    self.stack.append(new_style)
    self.emit_style(new_style)