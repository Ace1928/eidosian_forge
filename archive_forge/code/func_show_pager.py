import codecs
import numbers
import os
import platform
import re
import subprocess
import sys
from humanfriendly.compat import coerce_string, is_unicode, on_windows, which
from humanfriendly.decorators import cached
from humanfriendly.deprecation import define_aliases
from humanfriendly.text import concatenate, format
from humanfriendly.usage import format_usage
def show_pager(formatted_text, encoding=DEFAULT_ENCODING):
    """
    Print a large text to the terminal using a pager.

    :param formatted_text: The text to print to the terminal (a string).
    :param encoding: The name of the text encoding used to encode the formatted
                     text if the formatted text is a Unicode string (a string,
                     defaults to :data:`DEFAULT_ENCODING`).

    When :func:`connected_to_terminal()` returns :data:`True` a pager is used
    to show the text on the terminal, otherwise the text is printed directly
    without invoking a pager.

    The use of a pager helps to avoid the wall of text effect where the user
    has to scroll up to see where the output began (not very user friendly).

    Refer to :func:`get_pager_command()` for details about the command line
    that's used to invoke the pager.
    """
    if connected_to_terminal():
        command_line = get_pager_command(formatted_text)
        if which(command_line[0]):
            pager = subprocess.Popen(command_line, stdin=subprocess.PIPE)
            if is_unicode(formatted_text):
                formatted_text = formatted_text.encode(encoding)
            pager.communicate(input=formatted_text)
            return
    output(formatted_text)