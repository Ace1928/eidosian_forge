import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
@property
def next_elements(self):
    """All PageElements that were parsed after this one.

        :yield: A sequence of PageElements.
        """
    i = self.next_element
    while i is not None:
        yield i
        i = i.next_element