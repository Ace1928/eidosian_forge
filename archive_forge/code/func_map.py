import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
def map(self, function, kind):
    """Applies a function to the ``data`` element of events of ``kind`` in
        the selection.

        >>> import six
        >>> html = HTML('<html><head><title>Some Title</title></head>'
        ...               '<body>Some <em>body</em> text.</body></html>',
        ...             encoding='utf-8')
        >>> print(html | Transformer('head/title').map(six.text_type.upper, TEXT))
        <html><head><title>SOME TITLE</title></head><body>Some <em>body</em>
        text.</body></html>

        :param function: the function to apply
        :param kind: the kind of event the function should be applied to
        :rtype: `Transformer`
        """
    return self.apply(MapTransformation(function, kind))