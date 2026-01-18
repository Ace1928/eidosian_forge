import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
def empty(self):
    """Empty selected elements of all content.

        Example:

        >>> html = HTML('<html><head><title>Some Title</title></head>'
        ...             '<body>Some <em>body</em> text.</body></html>',
        ...             encoding='utf-8')
        >>> print(html | Transformer('.//em').empty())
        <html><head><title>Some Title</title></head><body>Some <em/>
        text.</body></html>

        :rtype: `Transformer`
        """
    return self.apply(EmptyTransformation())