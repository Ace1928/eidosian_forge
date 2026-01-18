import re
from humanfriendly.compat import HTMLParser, StringIO, name2codepoint, unichr
from humanfriendly.text import compact_empty_lines
from humanfriendly.terminal import ANSI_COLOR_CODES, ANSI_RESET, ansi_style
def parse_color(self, value):
    """
        Convert a CSS color to something that :func:`.ansi_style()` understands.

        :param value: A string like ``rgb(1,2,3)``, ``#AABBCC`` or ``yellow``.
        :returns: A color value supported by :func:`.ansi_style()` or :data:`None`.
        """
    if value.startswith('rgb'):
        tokens = re.findall('\\d+', value)
        if len(tokens) == 3:
            return tuple(map(int, tokens))
    elif value.startswith('#'):
        value = value[1:]
        length = len(value)
        if length == 6:
            return (int(value[:2], 16), int(value[2:4], 16), int(value[4:6], 16))
        elif length == 3:
            return (int(value[0], 16), int(value[1], 16), int(value[2], 16))
    value = value.lower()
    if value in ANSI_COLOR_CODES:
        return value