import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def encode_contents(self, indent_level=None, encoding=DEFAULT_OUTPUT_ENCODING, formatter='minimal'):
    """Renders the contents of this PageElement as a bytestring.

        :param indent_level: Each line of the rendering will be
           indented this many levels. (The formatter decides what a
           'level' means in terms of spaces or other characters
           output.) Used internally in recursive calls while
           pretty-printing.

        :param eventual_encoding: The bytestring will be in this encoding.

        :param formatter: A Formatter object, or a string naming one of
            the standard Formatters.

        :return: A bytestring.
        """
    contents = self.decode_contents(indent_level, encoding, formatter)
    return contents.encode(encoding)