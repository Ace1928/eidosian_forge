import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
def attr(self, name, value):
    """Add, replace or delete an attribute on selected elements.

        If `value` evaulates to `None` the attribute will be deleted from the
        element:

        >>> html = HTML('<html><head><title>Some Title</title></head>'
        ...             '<body>Some <em class="before">body</em> <em>text</em>.</body>'
        ...             '</html>', encoding='utf-8')
        >>> print(html | Transformer('body/em').attr('class', None))
        <html><head><title>Some Title</title></head><body>Some <em>body</em>
        <em>text</em>.</body></html>

        Otherwise the attribute will be set to `value`:

        >>> print(html | Transformer('body/em').attr('class', 'emphasis'))
        <html><head><title>Some Title</title></head><body>Some <em
        class="emphasis">body</em> <em class="emphasis">text</em>.</body></html>

        If `value` is a callable it will be called with the attribute name and
        the `START` event for the matching element. Its return value will then
        be used to set the attribute:

        >>> def print_attr(name, event):
        ...     attrs = event[1][1]
        ...     print(attrs)
        ...     return attrs.get(name)
        >>> print(html | Transformer('body/em').attr('class', print_attr))
        Attrs([(QName('class'), 'before')])
        Attrs()
        <html><head><title>Some Title</title></head><body>Some <em
        class="before">body</em> <em>text</em>.</body></html>

        :param name: the name of the attribute
        :param value: the value that should be set for the attribute.
        :rtype: `Transformer`
        """
    return self.apply(AttrTransformation(name, value))