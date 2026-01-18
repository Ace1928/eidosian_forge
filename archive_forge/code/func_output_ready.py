import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def output_ready(self, formatter=None):
    """Make this string ready for output by adding any subclass-specific
            prefix or suffix.

        :param formatter: A Formatter object, or a string naming one
            of the standard formatters. The string will be passed into the
            Formatter, but only to trigger any side effects: the return
            value is ignored.

        :return: The string, with any subclass-specific prefix and
           suffix added on.
        """
    if formatter is not None:
        ignore = self.format_string(self, formatter)
    return self.PREFIX + self + self.SUFFIX