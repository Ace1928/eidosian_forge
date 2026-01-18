import re
from humanfriendly.compat import HTMLParser, StringIO, name2codepoint, unichr
from humanfriendly.text import compact_empty_lines
from humanfriendly.terminal import ANSI_COLOR_CODES, ANSI_RESET, ansi_style
def html_to_ansi(data, callback=None):
    """
    Convert HTML with simple text formatting to text with ANSI escape sequences.

    :param data: The HTML to convert (a string).
    :param callback: Optional callback to pass to :class:`HTMLConverter`.
    :returns: Text with ANSI escape sequences (a string).

    Please refer to the documentation of the :class:`HTMLConverter` class for
    details about the conversion process (like which tags are supported) and an
    example with a screenshot.
    """
    converter = HTMLConverter(callback=callback)
    return converter(data)