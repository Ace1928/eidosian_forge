from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
def find_codec(self, charset):
    """Convert the name of a character set to a codec name.

        :param charset: The name of a character set.
        :return: The name of a codec.
        """
    value = self._codec(self.CHARSET_ALIASES.get(charset, charset)) or (charset and self._codec(charset.replace('-', ''))) or (charset and self._codec(charset.replace('-', '_'))) or (charset and charset.lower()) or charset
    if value:
        return value.lower()
    return None