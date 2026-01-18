import re
from humanfriendly.compat import HTMLParser, StringIO, name2codepoint, unichr
from humanfriendly.text import compact_empty_lines
from humanfriendly.terminal import ANSI_COLOR_CODES, ANSI_RESET, ansi_style
def emit_style(self, style=None):
    """
        Emit an ANSI escape sequence for the given or current style to the output stream.

        :param style: A dictionary with arguments for :func:`.ansi_style()` or
                      :data:`None`, in which case the style at the top of the
                      stack is emitted.
        """
    self.output.write(ANSI_RESET)
    style = self.current_style if style is None else style
    if style:
        self.output.write(ansi_style(**style))