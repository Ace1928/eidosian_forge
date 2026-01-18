import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
def apply(self, function):
    """Apply a transformation to the stream.

        Transformations can be chained, similar to stream filters. Any callable
        accepting a marked stream can be used as a transform.

        As an example, here is a simple `TEXT` event upper-casing transform:

        >>> def upper(stream):
        ...     for mark, (kind, data, pos) in stream:
        ...         if mark and kind is TEXT:
        ...             yield mark, (kind, data.upper(), pos)
        ...         else:
        ...             yield mark, (kind, data, pos)
        >>> short_stream = HTML('<body>Some <em>test</em> text</body>',
        ...                      encoding='utf-8')
        >>> print(short_stream | Transformer('.//em/text()').apply(upper))
        <body>Some <em>TEST</em> text</body>
        """
    transformer = Transformer()
    transformer.transforms = self.transforms[:]
    if isinstance(function, Transformer):
        transformer.transforms.extend(function.transforms)
    else:
        transformer.transforms.append(function)
    return transformer